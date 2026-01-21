from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional
# ============================================================
# Time utils (clean + reusable)
# ============================================================
def parse_dt(x: Any) -> Optional[datetime]:
    """Parse common date/datetime strings into datetime (best-effort)."""
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
def build_lab_timeline(lab_reports: list) -> list:
    tl = []
    for r in lab_reports or []:
        tl.append({
            "report_id": str(r.get("report_id")),
            "date": (r.get("report_date") or r.get("date") or "")[:10],
            "summary": r.get("summary", "")
        })
    return tl


def build_imaging_timeline(im_reports: list) -> list:
    tl = []
    for r in im_reports or []:
        imp = r.get("impression") or ""
        imp_short = (imp[:80] + "...") if len(imp) > 80 else imp
        tl.append({
            "report_id": str(r.get("report_id")),
            "date": (r.get("report_date") or r.get("date") or "")[:10],
            "modality": r.get("modality", ""),
            "summary": imp_short
        })
    return tl


def build_pathology_timeline(path_reports: list) -> list:
    tl = []
    for r in path_reports or []:
        summ = r.get("summary") or r.get("diagnosis") or ""
        if isinstance(summ, str) and len(summ) > 80:
            summ = summ[:80] + "..."
        tl.append({
            "report_id": str(r.get("report_id")),
            "date": (r.get("report_date") or r.get("date") or "")[:10],
            "summary": summ
        })
    return tl


def format_duration(seconds: float) -> str:
    """
    Convert seconds to human-readable format.
    
    Examples:
        format_duration(45.3) -> "45.3秒"
        format_duration(85) -> "1分25秒"
        format_duration(3661) -> "1小时1分1秒"
    
    Args:
        seconds: Duration in seconds (can be float)
    
    Returns:
        Human-readable duration string
    """
    if seconds < 0:
        return "0秒"
    
    total_seconds = int(seconds)
    milliseconds = int((seconds - total_seconds) * 10)  # One decimal place
    
    if total_seconds < 60:
        if milliseconds > 0:
            return f"{seconds:.1f}秒"
        return f"{total_seconds}秒"
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}小时")
    if minutes > 0:
        parts.append(f"{minutes}分")
    if secs > 0 or not parts:
        parts.append(f"{secs}秒")
    
    return "".join(parts)
