# clients/logger.py
import sqlite3
import json
from pathlib import Path


def _serialize_json_field(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


class DBLogger:
    def __init__(self, db_path="data/logs/omgs_api_trace.db"):
        db_file = Path(db_path)
        if db_file.parent != Path("."):
            db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_file)
        self.create_table()

    def create_table(self):
        # Check if table exists and has new columns
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='api_logs'
        """)
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # Create table with all columns
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                temperature REAL,
                seed INTEGER,
                input_text TEXT,
                output_text TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                raw_request TEXT,
                raw_response TEXT,
                latency_ms REAL,
                extra_body TEXT,
                reasoning_details TEXT,
                top_p REAL,
                max_completion_tokens INTEGER,
                reasoning_effort TEXT,
                raw_enabled INTEGER,
                status TEXT,
                error TEXT
            )
            """)
        else:
            # Add new columns if they don't exist (for backward compatibility)
            cursor.execute("PRAGMA table_info(api_logs)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if "provider" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN provider TEXT")
            if "seed" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN seed INTEGER")
            if "extra_body" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN extra_body TEXT")
            if "reasoning_details" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN reasoning_details TEXT")
            if "top_p" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN top_p REAL")
            if "max_completion_tokens" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN max_completion_tokens INTEGER")
            if "reasoning_effort" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN reasoning_effort TEXT")
            if "raw_enabled" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN raw_enabled INTEGER")
            if "status" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN status TEXT")
            if "error" not in columns:
                self.conn.execute("ALTER TABLE api_logs ADD COLUMN error TEXT")
        
        self.conn.commit()

    def log(self, **data):
        """data contains the above fields, including optional extra_body and reasoning_details"""
        self.conn.execute("""
            INSERT INTO api_logs (
                timestamp, provider, model, temperature, seed, input_text, output_text,
                input_tokens, output_tokens, total_tokens,
                raw_request, raw_response, latency_ms,
                extra_body, reasoning_details,
                top_p, max_completion_tokens, reasoning_effort,
                raw_enabled, status, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp"),
            data.get("provider") or "unknown",
            data.get("model"),
            data.get("temperature"),
            data.get("seed"),
            data.get("input_text"),
            data.get("output_text"),
            data.get("input_tokens"),
            data.get("output_tokens"),
            data.get("total_tokens"),
            _serialize_json_field(data.get("raw_request")),
            _serialize_json_field(data.get("raw_response")),
            data.get("latency_ms"),
            _serialize_json_field(data.get("extra_body")),
            _serialize_json_field(data.get("reasoning_details")),
            data.get("top_p"),
            data.get("max_completion_tokens"),
            data.get("reasoning_effort"),
            data.get("raw_enabled"),
            data.get("status"),
            data.get("error"),
        ))
        self.conn.commit()
