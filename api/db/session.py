import sqlite3
import os

DB_PATH = "jobs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Bảng jobs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        filename TEXT,
        status TEXT,
        message TEXT,
        folder TEXT,
        created_at DATETIME,
        training_status TEXT,
        training_message TEXT,
        completed_at DATETIME,
        result TEXT
    )
    """)
    
    # Bảng metrics
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        epoch INTEGER,
        loss REAL,
        psnr REAL,
        timestamp DATETIME,
        FOREIGN KEY (job_id) REFERENCES jobs (id)
    )
    """)
    
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Khởi tạo DB khi module được load
init_db()
