# aoai/logger.py
import sqlite3
import json

class DBLogger:
    def __init__(self, db_path="api_trace.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS api_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            temperature REAL,
            input_text TEXT,
            output_text TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            raw_request TEXT,
            raw_response TEXT,
            latency_ms REAL
        )
        """)
        self.conn.commit()

    def log(self, **data):
        """data 中包含上述字段"""
        self.conn.execute("""
            INSERT INTO api_logs (
                timestamp, model, temperature, input_text, output_text,
                input_tokens, output_tokens, total_tokens,
                raw_request, raw_response, latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp"),
            data.get("model"),
            data.get("temperature"),
            data.get("input_text"),
            data.get("output_text"),
            data.get("input_tokens"),
            data.get("output_tokens"),
            data.get("total_tokens"),
            json.dumps(data.get("raw_request")),
            json.dumps(data.get("raw_response")),
            data.get("latency_ms"),
        ))
        self.conn.commit()
