"""Pipeline statistics collection from API trace database."""

import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional, List


def collect_pipeline_stats(
    start_time: datetime,
    end_time: datetime,
    db_path: str
) -> Dict[str, Any]:
    """
    Collect pipeline execution statistics from API trace database.
    
    Queries the api_logs table for records between start_time and end_time,
    and aggregates token usage and model information.
    
    Args:
        start_time: Pipeline start timestamp
        end_time: Pipeline end timestamp
        db_path: Path to SQLite database file
    
    Returns:
        Dictionary containing:
        - total_seconds: Total execution time in seconds
        - total_input_tokens: Sum of all input tokens
        - total_output_tokens: Sum of all output tokens
        - total_tokens: Sum of all total tokens
        - models_used: List of unique models used
        - model_stats: List of per-model statistics (model, call_count, tokens)
    """
    stats = {
        "total_seconds": (end_time - start_time).total_seconds(),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
        "models_used": [],
        "model_stats": []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Convert datetime to string format for SQLite comparison
        # datetime.now() returns format like "2024-01-20 18:51:32.123456"
        # We use string comparison which works for ISO-like formats
        # For start_time, use exact format; for end_time, add 1 second to ensure we capture all records
        from datetime import timedelta
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        # Add 1 second to end_time to ensure we capture all records within the time range
        end_time_extended = end_time + timedelta(seconds=1)
        end_str = end_time_extended.strftime("%Y-%m-%d %H:%M:%S")
        
        # Query aggregated statistics by model
        # String comparison works correctly for ISO-like timestamp formats
        query = """
        SELECT 
            model,
            COUNT(*) as call_count,
            COALESCE(SUM(input_tokens), 0) as total_input_tokens,
            COALESCE(SUM(output_tokens), 0) as total_output_tokens,
            COALESCE(SUM(total_tokens), 0) as total_tokens
        FROM api_logs
        WHERE timestamp >= ? AND timestamp <= ?
        GROUP BY model
        ORDER BY total_tokens DESC
        """
        
        cursor.execute(query, (start_str, end_str))
        rows = cursor.fetchall()
        
        # Process results
        models_used = []
        model_stats = []
        
        for row in rows:
            model = row[0] or "unknown"
            call_count = row[1] or 0
            input_tokens = row[2] or 0
            output_tokens = row[3] or 0
            total_tokens = row[4] or 0
            
            models_used.append(model)
            model_stats.append({
                "model": model,
                "call_count": call_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            })
            
            # Aggregate totals
            stats["total_input_tokens"] += input_tokens
            stats["total_output_tokens"] += output_tokens
            stats["total_tokens"] += total_tokens
        
        stats["models_used"] = models_used
        stats["model_stats"] = model_stats
        
        conn.close()
        
    except Exception as e:
        # If database query fails, return stats with available data (time is always available)
        # Log error but don't fail the pipeline
        import sys
        print(f"[WARNING] Failed to collect pipeline statistics from database: {e}", file=sys.stderr)
    
    return stats
