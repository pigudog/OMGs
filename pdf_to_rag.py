#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified RAG Builder (Stable, Chroma v0.5+ compatible)

Commands:
  python pdf_to_rag.py build \
    --pdf_dir rag_pdf/chair \
    --out_dir rag_store/chair/corpus \
    --chunk_size 1200 \
    --chunk_overlap 200

  python pdf_to_rag.py index \
    --corpus_dir rag_store/chair/corpus \
    --index_dir rag_store/chair/index \
    --collection_name chair_chunks \
    --model BAAI/bge-m3 --device cpu

python pdf_to_rag.py search \
  --index_dir rag_store/chair/index/chroma \
  --collection_name chair_chunks \
  --model BAAI/bge-m3 \
  --device cpu \
  --query "NACT in ovarian cancer"

"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# NEW CHROMA API
from chromadb import PersistentClient


os.environ.setdefault("CHROMADB_TELEMETRY_ENABLED", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")


# ========================
# Utility Functions
# ========================

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def guess_doc_ids(pdf_path: Path) -> Dict[str, str]:
    stem = pdf_path.stem
    display = stem
    m = re.search(r"(v\d{8}|\d{4}[-_]\d{2}[-_]\d{2})", stem, re.I)
    version = m.group(1) if m else "v00000000"
    base = stem.replace(version, "").strip("._- ")
    doc_id = slugify(base) + "__" + slugify(version)
    return {"doc_id": doc_id, "display_name": display, "version": version}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _clean_metadata(md: Dict) -> Dict:
    out = {}
    if not isinstance(md, dict):
        return out
    for k, v in md.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
    return out


# ========================
# Data Structures
# ========================

@dataclass
class DocRegistryItem:
    doc_id: str
    display_name: str
    version: str
    path: str
    page_count: int


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    text: str
    section: Optional[str]
    page_from: Optional[int]
    page_to: Optional[int]
    display_name: Optional[str]
    version: Optional[str]
    created_at: Optional[str]
    hash: str


# ========================
# PDF → TXT
# ========================

def extract_pdf_to_txt(pdf_path: Path, out_txt: Path) -> int:
    doc = fitz.open(pdf_path)
    buf = [f"# {pdf_path.stem}\n"]

    for i, page in enumerate(doc):
        try:
            text = page.get_text("text")
        except Exception:
            text = ""
        text = (text or "").replace("\x0c", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        buf.append(f"\n---\n[PAGE {i+1}/{len(doc)}]\n\n{text}\n")

    full_text = "\n".join(buf)

    # Remove references
    full_text = re.split(r"\n\s*(References|参考文献)\b", full_text, flags=re.I)[0]

    out_txt.write_text(full_text, encoding="utf-8")
    return len(doc)


def _detect_section_heading(line: str) -> Optional[str]:
    s = line.strip()
    if s.startswith("#"):
        return s.lstrip("#").strip() or None
    if 0 < len(s) <= 80 and (s.isupper() or re.match(r"^[A-Z][A-Za-z0-9 \-:]{0,79}$", s)):
        return s
    return None


def split_txt_into_chunks(txt_str: str, meta: Dict,
                          chunk_size=1200, chunk_overlap=200,
                          created_at=None) -> Iterable[ChunkRecord]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n# ", "\n- ", "\n", " ", ""],
    )

    parts = txt_str.split("\n---\n")
    chunk_idx = 0
    last_section = None

    for part in parts:
        if not part.strip():
            continue
        content = part.strip()

        page_from = page_to = None
        if content.startswith("[PAGE"):
            head, _, body = content.partition("\n")
            m = re.search(r"\[PAGE\s+(\d+)", head)
            if m:
                page_from = page_to = int(m.group(1))
            content = body.strip()

        first_lines = "\n".join(content.splitlines()[:5])
        for ln in first_lines.splitlines():
            sec = _detect_section_heading(ln)
            if sec:
                last_section = sec
                break

        for ck in splitter.split_text(content):
            ck = ck.strip()
            if not ck:
                continue

            yield ChunkRecord(
                doc_id=meta["doc_id"],
                chunk_id=f"{chunk_idx:06d}",
                text=ck,
                section=last_section,
                page_from=page_from,
                page_to=page_to,
                display_name=meta.get("display_name"),
                version=meta.get("version"),
                created_at=created_at,
                hash=sha256_text(ck),
            )
            chunk_idx += 1
# ========================
# BUILD
# ========================

def cmd_build(args):
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    txt_dir = out_dir / "staging_txt"
    ck_dir = out_dir / "chunks"
    meta_dir = out_dir / "meta"

    for p in (txt_dir, ck_dir, meta_dir):
        p.mkdir(parents=True, exist_ok=True)

    registry_fp = meta_dir / "registry.jsonl"

    # load existing registry
    seen = {}
    if registry_fp.exists():
        for line in registry_fp.read_text("utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                seen[obj["doc_id"]] = DocRegistryItem(**obj)

    created_at_str = now_iso()
    pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))

    for pdf in pdf_files:
        ids = guess_doc_ids(pdf)
        txt_fp = txt_dir / f"{ids['doc_id']}.txt"
        page_count = extract_pdf_to_txt(pdf, txt_fp)

        txt = txt_fp.read_text("utf-8")
        ck_fp = ck_dir / f"{ids['doc_id']}.jsonl"

        with ck_fp.open("w", encoding="utf-8") as fw:
            for rec in split_txt_into_chunks(
                txt,
                ids,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                created_at=created_at_str,
            ):
                fw.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        seen[ids["doc_id"]] = DocRegistryItem(
            doc_id=ids["doc_id"],
            display_name=ids["display_name"],
            version=ids["version"],
            path=str(pdf.resolve()),
            page_count=page_count,
        )

    with registry_fp.open("w", encoding="utf-8") as fw:
        for it in seen.values():
            fw.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    print(f"✔ Build finished: TXT={txt_dir}, chunks={ck_dir}, registry={registry_fp}")


# ========================
# INDEX (W/ CHROMA v0.5+)
# ========================

def get_embeddings(model_name: str, device: str):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _iter_chunks(ck_dir: Path):
    for jf in sorted(ck_dir.glob("*.jsonl")):
        for line in jf.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            md = {k: v for k, v in obj.items() if k != "text"}
            yield text, md


def cmd_index(args):
    corpus_dir = Path(args.corpus_dir)
    ck_dir = corpus_dir / "chunks"
    index_dir = Path(args.index_dir)
    index_dir.mkdir(exist_ok=True, parents=True)

    print("✔ Initializing Chroma PersistentClient...")
    client = PersistentClient(path=str(index_dir))

    # Create collection if not exists
    existing = [c.name for c in client.list_collections()]
    if args.collection_name in existing:
        collection = client.get_collection(args.collection_name)
    else:
        collection = client.create_collection(
            name=args.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    print(f"✔ Using collection: {args.collection_name}")

    embed = get_embeddings(args.model, args.device)

    texts = []
    metadatas = []
    ids = []
    total = 0
    batch = 1024

    def flush():
        nonlocal texts, metadatas, ids, total
        if not texts:
            return
        embeddings = embed.embed_documents(texts)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        total += len(texts)
        texts.clear()
        metadatas.clear()
        ids.clear()

    for text, md in _iter_chunks(ck_dir):
        cid = md["doc_id"] + "::" + md["chunk_id"]
        texts.append(text)
        metadatas.append(_clean_metadata(md))
        ids.append(cid)

        if len(texts) >= batch:
            flush()

    flush()

    print(f"✔ Indexed {total} chunks into collection '{args.collection_name}'.")


# ========================
# SEARCH
# ========================

def cmd_search(args):
    index_dir = Path(args.index_dir)
    client = PersistentClient(path=str(index_dir))

    collection = client.get_collection(args.collection_name)
    embed = get_embeddings(args.model, args.device)

    q_emb = embed.embed_query(args.query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=args.topk,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\n🔎 Retrieved {len(docs)} chunks:\n")

    for i, (doc, md, dist) in enumerate(zip(docs, metas, dists), 1):
        doc_id = md.get("doc_id", "")
        p = md.get("page_from")
        snippet = doc.replace("\n", " ")[:300]

        print(f"[{i}] score={dist:.4f}  {doc_id}  [PAGE {p}]")
        print("   ", snippet, "…\n")


# ========================
# CLI
# ========================

def main():
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- BUILD ----
    pb = sub.add_parser("build")
    pb.add_argument("--pdf_dir", required=True)
    pb.add_argument("--out_dir", required=True)
    pb.add_argument("--chunk_size", type=int, default=1200)
    pb.add_argument("--chunk_overlap", type=int, default=200)
    pb.set_defaults(func=cmd_build)

    # ---- INDEX ----
    pi = sub.add_parser("index")
    pi.add_argument("--corpus_dir", required=True)
    pi.add_argument("--index_dir", required=True)
    pi.add_argument("--collection_name", required=True)
    pi.add_argument("--model", default="BAAI/bge-m3")
    pi.add_argument("--device", default="cpu")
    pi.set_defaults(func=cmd_index)

    # ---- SEARCH ----
    ps = sub.add_parser("search")
    ps.add_argument("--index_dir", required=True)
    ps.add_argument("--collection_name", required=True)
    ps.add_argument("--model", default="BAAI/bge-m3")
    ps.add_argument("--device", default="cpu")
    ps.add_argument("--query", required=True)
    ps.add_argument("--topk", type=int, default=4)
    ps.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
