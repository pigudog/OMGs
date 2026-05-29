from datetime import datetime, date, timedelta
import re
from typing import Any, Dict, List, Optional
# ============================================================
# Time utils (clean + reusable)
# ============================================================
def parse_dt(x: Any) -> Optional[datetime]:
    """Parse common date/datetime strings into datetime (best-effort).

    Supports formats like:
    - "2022-12-09"
    - "2022-12-09T00:00:00"
    - "2023-06-05T15:12:06"
    - "2022-12" (year-month, treated as first day of month)
    - "2022" (year only, treated as Jan 1)

    Returns:
        datetime object or None if parsing fails
    """
    if not x:
        return None
    s = str(x).strip().replace("/", "-")

    # Strip common timezone suffixes (e.g., Z, +08:00) for fromisoformat
    s_clean = s.replace("Z", "").split("+")[0].strip()

    # Try full ISO first (keeps microseconds if present)
    try:
        return datetime.fromisoformat(s_clean.replace("T", " "))
    except Exception:
        pass

    # Fallback: truncate to second / date only
    for cand in (s_clean[:19], s_clean[:10]):
        try:
            return datetime.fromisoformat(cand.replace("T", " "))
        except Exception:
            pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s_clean[:19], fmt)
        except Exception:
            pass

    # Handle incomplete dates: "2022-12" (year-month) or "2022" (year)
    # Year-month format: treat as first day of month
    if re.match(r'^\d{4}-\d{2}$', s_clean):
        try:
            return datetime.strptime(s_clean + "-01", "%Y-%m-%d")
        except Exception:
            pass

    # Year only format: treat as January 1st
    if re.match(r'^\d{4}$', s_clean):
        try:
            return datetime.strptime(s_clean + "-01-01", "%Y-%m-%d")
        except Exception:
            pass

    return None


def parse_date(x: Any) -> Optional[date]:
    """
    Parse date-like strings and return date object (not datetime).
    Supports formats like:
    - "2022-12-09"
    - "2022-12-09T00:00:00"
    - "2023-06-05T15:12:06"
    
    Returns:
        date object or None if parsing fails
    """
    dt = parse_dt(x)
    return dt.date() if dt is not None else None


def make_cutoff(index_time: Optional[str], days_after: int = 1) -> Optional[datetime]:
    """index_time + days_after (for hard filter of reports)."""
    t = parse_dt(index_time)
    return None if t is None else (t + timedelta(days=days_after))


def filter_before(items: List[Dict[str, Any]], key: str, cutoff_dt: Optional[datetime]) -> List[Dict[str, Any]]:
    """Keep rows with key <= cutoff_dt.

    If a row has no parsable datetime, keep it (do not drop unknown-date reports).
    Also try common fallback keys: 'date' and 'time'.
    """
    if cutoff_dt is None:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        dt = parse_dt(it.get(key)) or parse_dt(it.get("date")) or parse_dt(it.get("time"))
        # If cannot parse date, keep the item to avoid filtering everything out.
        if dt is None:
            out.append(it)
            continue
        if dt <= cutoff_dt:
            out.append(it)
    return out


def safe_date10(x: Any) -> Optional[str]:
    dt = parse_dt(x)
    return None if dt is None else dt.date().isoformat()


def report_range(reports, key: str = "report_date") -> str:
    dts = [parse_dt(r.get(key)) for r in reports]
    dts = [d for d in dts if d is not None]
    if not dts:
        return "no parsable dates"
    return f"{min(dts)} ~ {max(dts)} (n={len(dts)})"

###############################################################################
# 🔧 Timeline rebuilders
###############################################################################
def _truncate(s: str, max_len: int = 80) -> str:
    """Truncate string with ellipsis if needed."""
    if not isinstance(s, str):
        return ""
    return (s[:max_len] + "...") if len(s) > max_len else s


def _build_report_timeline(
    reports: list,
    *,
    summary_field: str,
    fallback_summary_fields: list = None,
    extra_fields: dict = None,
    truncate_summary: bool = False,
) -> list:
    """Generic timeline builder for any report type.

    Args:
        reports: List of report dictionaries
        summary_field: Primary field to use for summary
        fallback_summary_fields: List of fallback fields if primary is empty
        extra_fields: Additional fields to include (name -> field_key)
        truncate_summary: Whether to truncate summary to 80 chars
    """
    tl = []
    fallback_summary_fields = fallback_summary_fields or []
    extra_fields = extra_fields or {}

    for r in reports or []:
        # Get summary with fallback fields
        summ = r.get(summary_field, "")
        if not summ:
            for f in fallback_summary_fields:
                summ = r.get(f, "")
                if summ:
                    break

        if truncate_summary:
            summ = _truncate(summ)

        item = {
            "report_id": str(r.get("report_id")),
            "date": (r.get("report_date") or r.get("date") or "")[:10],
            "summary": summ,
        }
        # Add any extra fields
        for name, field_key in extra_fields.items():
            item[name] = r.get(field_key, "")
        tl.append(item)
    return tl


def build_lab_timeline(lab_reports: list) -> list:
    """Build timeline from lab reports."""
    return _build_report_timeline(
        lab_reports,
        summary_field="summary",
    )


def build_imaging_timeline(im_reports: list) -> list:
    """Build timeline from imaging reports."""
    return _build_report_timeline(
        im_reports,
        summary_field="impression",
        extra_fields={"modality": "modality"},
        truncate_summary=True,
    )


def build_pathology_timeline(path_reports: list) -> list:
    """Build timeline from pathology reports."""
    return _build_report_timeline(
        path_reports,
        summary_field="summary",
        fallback_summary_fields=["diagnosis"],
        truncate_summary=True,
    )


def format_duration(seconds: float) -> str:
    """
    Convert seconds to human-readable format.

    Examples:
        format_duration(45.3) -> "45.3s"
        format_duration(85) -> "1m 25s"
        format_duration(3661) -> "1h 1m 1s"

    Args:
        seconds: Duration in seconds (can be float)

    Returns:
        Human-readable duration string
    """
    if seconds < 0:
        return "0s"

    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 10)  # One decimal place

    if total_seconds < 60:
        if milliseconds > 0:
            return f"{seconds:.1f}s"
        return f"{total_seconds}s"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def newest_date(items: list) -> str:
    """Get the newest date from a list of report dicts.

    Args:
        items: List of dictionaries containing date/report_date/time keys

    Returns:
        Formatted date string (YYYY-MM-DD) or "-" if no valid dates
    """
    dts = []
    for r in items or []:
        dt = parse_dt(r.get("date")) or parse_dt(r.get("report_date")) or parse_dt(r.get("time"))
        if dt is not None:
            dts.append(dt)
    return (max(dts).strftime("%Y-%m-%d") if dts else "-")
