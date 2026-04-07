"""
Data Manager - SQLite database & CSV management for TinNam data.
Handles storage, retrieval, validation and deduplication.
"""
import sqlite3
import csv
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'tinnam_data.db')
CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def get_db():
    """Get database connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_db()
    c = conn.cursor()
    
    # Mega 6/45 table
    c.execute('''CREATE TABLE IF NOT EXISTS mega645 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        jackpot TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(draw_date, n1, n2, n3, n4, n5, n6)
    )''')
    
    # Power 6/55 table
    c.execute('''CREATE TABLE IF NOT EXISTS power655 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL,
        n1 INTEGER NOT NULL,
        n2 INTEGER NOT NULL,
        n3 INTEGER NOT NULL,
        n4 INTEGER NOT NULL,
        n5 INTEGER NOT NULL,
        n6 INTEGER NOT NULL,
        bonus INTEGER NOT NULL,
        jackpot TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(draw_date, n1, n2, n3, n4, n5, n6, bonus)
    )''')
    
    conn.commit()
    conn.close()
    print(f"[DB] Database initialized at {DB_PATH}")


def insert_mega645(rows):
    """Insert Mega 6/45 results. rows = list of (date, n1..n6, jackpot)."""
    conn = get_db()
    c = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO mega645 
                        (draw_date, n1, n2, n3, n4, n5, n6, jackpot) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', row)
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[DB] Error inserting Mega row: {e}")
    conn.commit()
    conn.close()
    print(f"[DB] Mega 6/45: Inserted {inserted}/{len(rows)} rows")
    return inserted


def insert_power655(rows):
    """Insert Power 6/55 results. rows = list of (date, n1..n6, bonus, jackpot)."""
    conn = get_db()
    c = conn.cursor()
    inserted = 0
    for row in rows:
        try:
            c.execute('''INSERT OR IGNORE INTO power655 
                        (draw_date, n1, n2, n3, n4, n5, n6, bonus, jackpot) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)
            if c.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"[DB] Error inserting Power row: {e}")
    conn.commit()
    conn.close()
    print(f"[DB] Power 6/55: Inserted {inserted}/{len(rows)} rows")
    return inserted


def get_mega645_all():
    """Get all Mega 6/45 results sorted by date."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM mega645 ORDER BY draw_date ASC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_power655_all():
    """Get all Power 6/55 results sorted by date."""
    conn = get_db()
    rows = conn.execute(
        'SELECT * FROM power655 ORDER BY draw_date ASC'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_mega645_numbers():
    """Get Mega 6/45 numbers only as list of lists [[n1..n6], ...]."""
    rows = get_mega645_all()
    return [[r['n1'], r['n2'], r['n3'], r['n4'], r['n5'], r['n6']] for r in rows]


def get_power655_numbers():
    """Get Power 6/55 numbers only as list of lists [[n1..n6, bonus], ...]."""
    rows = get_power655_all()
    return [[r['n1'], r['n2'], r['n3'], r['n4'], r['n5'], r['n6'], r['bonus']] for r in rows]


def get_latest_date(lottery_type):
    """Get the latest draw date for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT MAX(draw_date) as max_date FROM {table}').fetchone()
    conn.close()
    return row['max_date'] if row else None


def get_count(lottery_type):
    """Get total number of draws for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT COUNT(*) as cnt FROM {table}').fetchone()
    conn.close()
    return row['cnt']


def get_first_date(lottery_type):
    """Get the earliest draw date for a lottery type."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    row = conn.execute(f'SELECT MIN(draw_date) as min_date FROM {table}').fetchone()
    conn.close()
    return row['min_date'] if row else None


def export_csv(lottery_type):
    """Export data to CSV file."""
    if lottery_type == 'mega':
        rows = get_mega645_all()
        filename = os.path.join(CSV_DIR, 'mega645.csv')
        headers = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'jackpot']
    else:
        rows = get_power655_all()
        filename = os.path.join(CSV_DIR, 'power655.csv')
        headers = ['draw_date', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'bonus', 'jackpot']
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[CSV] Exported {len(rows)} rows to {filename}")
    return filename


def get_recent(lottery_type, n=20):
    """Get N most recent draws."""
    conn = get_db()
    table = 'mega645' if lottery_type == 'mega' else 'power655'
    rows = conn.execute(
        f'SELECT * FROM {table} ORDER BY draw_date DESC LIMIT ?', (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Auto-init on import
init_db()
