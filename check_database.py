import sqlite3
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent

database_paths = [
    project_dir / "data" / "inventory.db",
    project_dir / "src" / "inventory_database.db",
]

for db_path in database_paths:
    print(f"\nChecking: {db_path}")
    print("File exists:", db_path.is_file())

    if not db_path.is_file():
        continue

    with sqlite3.connect(
        f"{db_path.as_uri()}?mode=ro",
        uri=True
    ) as conn:
        tables = conn.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
        """).fetchall()

        print("Tables:", [table[0] for table in tables])