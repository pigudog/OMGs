#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_ehr.py version 3.2 (EHR-only, minimal-change hardening)

Key changes:
- Removed scene classification/routing (no longer needed).
- Real retry: retries on API error OR empty output, with exponential backoff + jitter.
- Compact JSONL output (single line, no spaces) for large-scale processing.
- Stream input reading (supports large JSONL) while still showing total progress.
- Patient-id sanitization for safe filenames.
- Separate audit_ehr with token/finish/attempt/error tracking, plus optional JSON repair audit.
- Force English for EHR extraction by default.
- Removed `question` alias from outputs, and `ehr_extracted_text` is now optional (default off).
"""

import os
import json
import time
import argparse
import sys
import random
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from aoai import OpenAIWrapper


# ===========================
# Logging helpers
# ===========================

def _log(msg: str, *, verbose: bool = False):
    """Print logs only when verbose is enabled."""
    if verbose:
        print(msg)


# ===========================
# Message helpers
# ===========================

def build_messages(system_prompt: str, user_text: str):
    """Build Chat Completions message payload."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def prepend_document_time(src: str, time_val: Any) -> str:
    """Prefix source text with DOCUMENT_TIME if time_val is provided."""
    if not time_val:
        return src
    doc_time = str(time_val).strip()
    if not doc_time:
        return src
    return f"DOCUMENT_TIME: {doc_time}\n\n" + src


# ===========================
# Client initialization
# ===========================
def init_client(db_path: str = "api_trace.db") -> OpenAIWrapper:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not endpoint or not api_key:
        raise RuntimeError("缺少 AZURE_OPENAI_ENDPOINT 或 AZURE_OPENAI_API_KEY 环境变量。")

    print(f"[init_client] endpoint: {endpoint}")
    print(f"[init_client] api_key: {'*'*len(api_key)}")

    return OpenAIWrapper(api_key=api_key, base_url=endpoint, db_path=db_path)


# ===========================
# Load prompt.json / scene.json
# ===========================
def load_prompt_config(prompt_path: str) -> Dict[str, Any]:
    data = json.loads(Path(prompt_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prompts" in data:
        data = data["prompts"]
    if "MASTER_INSTRUCTIONS" not in data or "OUTPUT_SCHEMA" not in data:
        raise RuntimeError(f"{prompt_path} 不符合 schema-driven 格式")
    return data


def compact_schema_shape(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: compact_schema_shape(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return [compact_schema_shape(schema[0])] if schema else []
    return schema


def build_system_prompt(cfg: Dict[str, Any], force_english: bool = False) -> str:
    master = cfg["MASTER_INSTRUCTIONS"].strip()
    schema_shape = compact_schema_shape(cfg["OUTPUT_SCHEMA"])
    final_req = cfg["FINAL_OUTPUT_REQUIREMENT"].strip()

    if force_english:
        master += (
            "\n\nLANGUAGE REQUIREMENT:\n"
            "- All free-text values MUST be in English.\n"
            "- Drugs/regimens/doses remain EXACTLY as source.\n"
        )

    schema_text = json.dumps(schema_shape, ensure_ascii=False)
    return master + "\n\nOUTPUT_SCHEMA_SHAPE:\n" + schema_text + "\n\n" + final_req


# ===========================
# API call + retry mechanism
# ===========================

def call_extract_once(client, deployment, system_prompt, emr_text, *, max_completion_tokens):
    """One-shot call to Azure OpenAI / OpenAI Chat Completions.

    Returns:
        content, req_id, err, tokens_used, finish_reason
    """
    messages = build_messages(system_prompt, emr_text)
    try:
        resp = client.chat_completion(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        content = (resp.choices[0].message.content or "").strip()
        req_id = getattr(resp, "id", "")
        usage = getattr(resp, "usage", None)
        tokens_used = getattr(usage, "total_tokens", 0) if usage else 0
        finish = getattr(resp.choices[0], "finish_reason", "")
        return content, req_id, None, tokens_used, finish
    except Exception as e:
        return "", "", str(e), 0, ""



def call_with_retry(func, *args, max_retries=4, delay=2, max_delay=30, verbose: bool = False, **kwargs):
    """
    Retry wrapper that works with call_extract_once-style return tuple.
    Retries when:
      - err is not None
      - content is empty (treated as failure)
    """
    errors: List[str] = []
    last = ("", "", "retry_failed", 0, "")

    for attempt in range(max_retries):
        content, req_id, err, tokens, finish = func(*args, **kwargs)
        last = (content, req_id, err, tokens, finish)

        if err is None and (content or "").strip():
            return content, req_id, None, tokens, finish, errors, attempt + 1

        # record error
        if err:
            errors.append(err)
        else:
            errors.append("empty_output")

        # backoff
        sleep_s = min(max_delay, delay * (2 ** attempt)) + random.random()
        _log(f"[retry] attempt={attempt+1}/{max_retries} err={errors[-1]} sleep={sleep_s:.2f}s", verbose=verbose)
        time.sleep(sleep_s)

    # exhausted
    content, req_id, err, tokens, finish = last
    if err is None and not (content or "").strip():
        err = "empty_output"
    return content, req_id, err or "retry_failed", tokens, finish, errors, max_retries


# ===========================
# JSON parsing / repairing
# ===========================
def extract_json_object(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    return s[start:end + 1] if start != -1 and end != -1 else ""


def try_parse_json(text: str):
    if not text:
        return None, "empty_output"
    try:
        obj = json.loads(extract_json_object(text))
        return obj, "ok"
    except Exception:
        return None, "json_decode_error"


def repair_json_once(client, deployment, bad_json_text, *, max_completion_tokens):
    sys_prompt = (
        "You are a JSON repair engine.\n"
        "Return ONLY valid JSON.\n"
        "No markdown. No explanation."
    )
    user = "Fix to valid JSON:\n\n" + bad_json_text
    return call_extract_once(
        client,
        deployment,
        sys_prompt,
        user,
        max_completion_tokens=max_completion_tokens,
    )


# ===========================
# Small utilities (P2)
# ===========================
def sanitize_patient_id(x: Any, max_len: int = 60) -> str:
    """
    Make patient_id safe for filenames and logs.
    """
    if x is None:
        s = ""
    elif isinstance(x, str):
        s = x
    else:
        try:
            s = json.dumps(x, ensure_ascii=False)
        except Exception:
            s = str(x)

    s = s.strip()
    if not s:
        return "UNKNOWN"

    # replace illegal filename chars and whitespace
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s)
    s = s.strip("._-")
    return (s[:max_len] or "UNKNOWN")


def compact_json_line(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


# ===========================
# Helper: strip internal fields
# ===========================
def strip_internal_fields(ehr: Any) -> Any:
    """Remove non-factual / internal control fields to keep outputs compact."""
    if not isinstance(ehr, dict):
        return ehr

    # TIMELINE.constraints
    try:
        timeline = ehr.get("CASE_CORE", {}).get("TIMELINE", None)
        if isinstance(timeline, dict):
            timeline.pop("constraints", None)
    except Exception:
        pass

    # MED_ONC.notes.allowed_content
    try:
        med_onc = ehr.get("MED_ONC", None)
        if isinstance(med_onc, dict):
            notes = med_onc.get("notes", None)
            if isinstance(notes, dict):
                notes.pop("allowed_content", None)
                # If notes becomes empty, drop it
                if not notes:
                    med_onc.pop("notes", None)
    except Exception:
        pass

    # LAB_TRENDS.rules
    try:
        lab_trends = ehr.get("LAB_TRENDS", None)
        if isinstance(lab_trends, dict):
            lab_trends.pop("rules", None)
    except Exception:
        pass

    return ehr


# ===========================
# Main processing pipeline
# ===========================
def process_file(
    client,
    deployment,
    input_path,
    output_path,
    prompt_path,
    *,
    max_completion_tokens,
    retry,
    input_field,
    enable_json_repair,
    txt_dir,
    verbose: bool = False,
):
    print(f"🚀 begin: {input_path}")
    # Load EHR prompt config
    cfg_ehr = load_prompt_config(prompt_path)
    system_ehr = build_system_prompt(cfg_ehr, force_english=True)  # always English for EHR extraction

    # Count total lines (keep tqdm total, but stream processing)
    with open(input_path, "r", encoding="utf-8") as fr:
        total = sum(1 for _ in fr)

    print(f"📌 Total records in input file: {total}")

    # Ensure output dirs
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if txt_dir:
        Path(txt_dir).mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fw, open(input_path, "r", encoding="utf-8") as fr:
        pbar = tqdm(total=total, ncols=120, desc="Processing")

        for idx, line in enumerate(fr, start=1):
            line = (line or "").strip()
            if not line:
                pbar.update(1)
                continue

            # Load per-line JSONL
            try:
                row = json.loads(line)
            except Exception:
                out_err = {
                    "patient_id": "PARSE_ERROR",
                    "Time": "",
                    "ehr_extracted": None,
                    "audit_ehr": {
                        "tokens": 0,
                        "finish": "",
                        "parse_status": "input_json_decode_error",
                        "attempts": 0,
                        "errors": ["input_json_decode_error"],
                        "req_id": "",
                        "err": "JSONDecodeError",
                        "repair_attempted": False,
                        "repair_used": False,
                        "repair_parse_status": None,
                        "repair_raw": None,
                    },
                    "error": "JSONDecodeError",
                    "source_text": line,
                }
                fw.write(compact_json_line(out_err) + "\n")
                pbar.update(1)
                continue

            # Retrieve input text (question / EMR content)
            src = row.get(input_field, "")
            if not isinstance(src, str):
                src = json.dumps(src, ensure_ascii=False)

            # Identify patient ID from various possible fields
            raw_patient_id = row.get("meta_info") or row.get("patient_id") or row.get("name") or ""
            patient_id = raw_patient_id if isinstance(raw_patient_id, str) else raw_patient_id
            patient_id_safe = sanitize_patient_id(raw_patient_id)
            Time_val = row.get("Time") or ""

            # Add DOCUMENT_TIME prefix (if available) to help the model resolve visit timing
            src_for_llm = prepend_document_time(src, Time_val)

            pbar.set_description(f"{idx}/{total} | {patient_id_safe}")

            # -------------------------------
            # 1) MAIN EHR EXTRACTION
            # -------------------------------
            ehr_content, req_id, err, tokens, finish, errs, attempts = call_with_retry(
                call_extract_once,
                client,
                deployment,
                system_ehr,
                src_for_llm,
                max_retries=retry,
                max_completion_tokens=max_completion_tokens,
                verbose=verbose,
            )

            parsed, status = try_parse_json(ehr_content)

            repair_attempted = False
            repair_used = False
            repair_parse_status = None
            repair_raw = None

            # Remove internal fields if parsed
            if parsed is not None:
                parsed = strip_internal_fields(parsed)

            # Attempt repair if JSON broken
            if parsed is None and enable_json_repair and (ehr_content or "").strip():
                repair_attempted = True
                rep_cont, rep_req, rep_err, rep_tokens, rep_finish = repair_json_once(
                    client, deployment, ehr_content, max_completion_tokens=max_completion_tokens
                )
                repair_raw = (rep_cont or "").strip() if rep_cont else None
                parsed2, status2 = try_parse_json(rep_cont)
                repair_parse_status = status2

                if parsed2 is not None:
                    parsed2 = strip_internal_fields(parsed2)
                    parsed = parsed2
                    status = status2
                    ehr_content = rep_cont
                    repair_used = True

            # -------------------------------
            # Write output JSONL (compact + ordered)
            #
            # Naming convention (as requested):
            # - question: final structured EHR JSON (dict) used by downstream agents
            # - question_raw: original unstructured source text that was fed into the extractor
            # -------------------------------
            out = dict(row)

            # Remove keys if already exist, so re-inserting will enforce order
            for k in [
                "patient_id", "Time",
                "question", "question_raw",
                "ehr_extracted", "ehr_extracted_text",  # legacy keys
                "audit_ehr",
                "source_text",  # legacy key
            ]:
                out.pop(k, None)

            # Insert in desired order
            out["patient_id"] = patient_id
            out["Time"] = Time_val

            # Primary outputs
            out["question"] = parsed
            out["question_raw"] = src

            # Optional: keep legacy keys OFF by default to avoid duplication.
            # If you need back-compat later, you can add flags to include them.

            out["audit_ehr"] = {
                "tokens": tokens,
                "finish": finish,
                "parse_status": status,
                "attempts": attempts,
                "errors": errs,
                "req_id": req_id,
                "err": err,
                "repair_attempted": repair_attempted,
                "repair_used": repair_used,
                "repair_parse_status": repair_parse_status,
                "repair_raw": repair_raw,
            }

            fw.write(compact_json_line(out) + "\n")

            # -------------------------------
            # Optional: TXT preview output
            # -------------------------------
            if txt_dir:
                txt_path = Path(txt_dir) / f"{idx:06d}_{patient_id_safe}.txt"
                with open(txt_path, "w", encoding="utf-8") as ftxt:
                    ftxt.write(f"patient_id: {patient_id}\n")
                    ftxt.write(f"patient_id_safe: {patient_id_safe}\n")
                    ftxt.write(f"Time: {Time_val}\n\n")

                    ftxt.write("==== SOURCE TEXT ====\n")
                    ftxt.write(src + "\n\n")

                    ftxt.write("==== EHR ====\n")
                    ftxt.write(f"finish: {finish}\n")
                    ftxt.write(f"parse_status: {status}\n")
                    ftxt.write(f"attempts: {attempts}\n")
                    if errs:
                        ftxt.write("errors:\n" + "\n".join([f"- {e}" for e in errs]) + "\n")
                    ftxt.write(f"repair_attempted: {repair_attempted}\n")
                    ftxt.write(f"repair_used: {repair_used}\n")
                    if repair_parse_status:
                        ftxt.write(f"repair_parse_status: {repair_parse_status}\n")
                    ftxt.write("\n")

                    ftxt.write("==== EHR JSON ====\n")
                    if parsed is not None:
                        ftxt.write(json.dumps(parsed, ensure_ascii=False, indent=2))
                    else:
                        ftxt.write((ehr_content or "").strip())

            pbar.update(1)

        pbar.close()

    print(f"✅ 已写入 {output_path}")



# ===========================
# CLI
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--field", default="question")
    ap.add_argument("--max-completion-tokens", type=int, default=40000)
    ap.add_argument("--retries", type=int, default=4)
    # Removed log_file argument
    ap.add_argument("--db_path", default="api_trace.db")
    ap.add_argument("--disable-json-repair", action="store_true")
    ap.add_argument("--txt-dir", default="")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose retry logs")
    ap.add_argument("--quiet", action="store_true", help="Suppress retry logs (default)")

    args = ap.parse_args()

    client = init_client(db_path=args.db_path)

    process_file(
        client=client,
        deployment=args.deployment,
        input_path=args.input,
        output_path=args.output,
        prompt_path=args.prompts,
        max_completion_tokens=args.max_completion_tokens,
        retry=args.retries,
        input_field=args.field,
        enable_json_repair=not args.disable_json_repair,
        txt_dir=args.txt_dir,
        verbose=(args.verbose and not args.quiet),
    )


if __name__ == "__main__":
    main()
