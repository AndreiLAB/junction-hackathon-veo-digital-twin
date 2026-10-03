import sqlite3
import numpy as np
import io
from config import DB_PATH

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guid TEXT UNIQUE,
        name TEXT,
        x REAL,
        y REAL,
        z REAL,
        qw REAL,
        qx REAL,
        qy REAL,
        qz REAL
    );

    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guid TEXT UNIQUE,
        scan_id INTEGER,
        name TEXT,
        file_path TEXT,
        width INTEGER,
        height INTEGER,
        x REAL,
        y REAL,
        z REAL,
        qw REAL,
        qx REAL,
        qy REAL,
        qz REAL,
        forward_x REAL,
        forward_y REAL,
        forward_z REAL,
        yaw REAL,
        pitch REAL,
        roll REAL,
        projection_type TEXT,
        FOREIGN KEY (scan_id) REFERENCES scans(id)
    );

    CREATE TABLE IF NOT EXISTS embeddings (
        image_id INTEGER,
        model_name TEXT,
        embedding array,
        PRIMARY KEY (image_id, model_name),
        FOREIGN KEY (image_id) REFERENCES images(id)
    );

    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        class_name TEXT,
        confidence REAL,
        bbox TEXT,
        ocr_text TEXT,
        ocr_conf REAL,
        FOREIGN KEY (image_id) REFERENCES images(id)
    );

    CREATE TABLE IF NOT EXISTS assets (
        asset_id TEXT PRIMARY KEY,
        cubicle_id TEXT,
        type TEXT,
        manufacturer TEXT,
        model TEXT
    );

    CREATE TABLE IF NOT EXISTS tags (
        tag_id TEXT PRIMARY KEY,
        parent_tag_id TEXT,
        label TEXT,
        asset_type TEXT,
        confidence REAL,
        x REAL,
        y REAL,
        z REAL,
        source_image_id INTEGER,
        source_scan_id INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_images_scan_id ON images(scan_id);
    CREATE INDEX IF NOT EXISTS idx_images_guid ON images(guid);
    CREATE INDEX IF NOT EXISTS idx_scans_guid ON scans(guid);
    """)

    conn.commit()
    conn.close()
