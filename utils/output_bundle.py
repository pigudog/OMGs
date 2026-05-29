"""Helpers for creating clean, auditable run output bundles."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RUN_RE_TEMPLATE = r"^{input_name}_run(\d{{3}})_\d{{8}}_\d{{6}}$"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_input_name(input_path: str) -> str:
    """Return a filesystem-safe run prefix derived from the input filename."""

    name = Path(input_path).stem.strip() or "input"
    name = _SAFE_NAME_RE.sub("_", name).strip("._-")
    return name or "input"


def sha256_file(path: str) -> Optional[str]:
    file_path = Path(path)
    if not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_input_path(path: str, base_dir: Path | None = None) -> str:
    """Avoid storing host-specific absolute paths when a relative path is enough."""

    input_path = Path(path)
    if not input_path.is_absolute():
        return path

    if base_dir is not None:
        try:
            return str(input_path.relative_to(base_dir.resolve()))
        except ValueError:
            pass
    return input_path.name


def next_run_dir(output_root: str | Path, input_name: str, now: datetime | None = None) -> tuple[str, Path]:
    """Create and return the next per-input run directory.

    The run counter is scoped to ``input_name``. Directory creation is retried to
    avoid collisions if two runs start at nearly the same time.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or utc_now()).strftime("%Y%m%d_%H%M%S")

    pattern = re.compile(_RUN_RE_TEMPLATE.format(input_name=re.escape(input_name)))
    existing_indices = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=0) + 1
    while True:
        run_id = f"{input_name}_run{next_index:03d}_{timestamp}"
        run_dir = root / run_id
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            next_index += 1


def extract_case_id(sample: Dict[str, Any], input_index: int) -> str:
    """Return the de-identified case identifier used in result bundles."""

    value = sample.get("meta_info")
    if value not in (None, ""):
        return str(value)

    summary = sample.get("summary")
    if isinstance(summary, dict):
        for key in ("Sample_id", "sample_id", "case_id", "Case_id"):
            value = summary.get(key)
            if value not in (None, ""):
                return str(value)

    for key in ("case_id", "Case_id", "sample_id"):
        value = sample.get(key)
        if value not in (None, ""):
            return str(value)

    return f"input_index_{input_index}"


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_manifest(
    *,
    run_id: str,
    input_path: str,
    input_count: int,
    requested_samples: int,
    agent: str,
    model: str,
    provider: str,
    output_files: Iterable[str],
    created_at: datetime | None = None,
    base_dir: Path | None = None,
) -> Dict[str, Any]:
    created = created_at or utc_now()
    input_name = sanitize_input_name(input_path)
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": created.isoformat(),
        "input_name": input_name,
        "input_path": display_input_path(input_path, base_dir=base_dir),
        "input_sha256": sha256_file(input_path),
        "input_record_count": input_count,
        "requested_samples": requested_samples,
        "agent": agent,
        "model": model,
        "provider": provider,
        "output_files": list(output_files),
        "privacy": {
            "scope": "output_answer_bundle",
            "contains_raw_input": False,
            "contains_raw_prompts": False,
            "contains_raw_model_logs": False,
            "contains_sqlite_logs": False,
        },
    }
