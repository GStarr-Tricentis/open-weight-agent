import sqlite3

from graph_pipeline.loaders.base import DataLoader


class SqlLoader(DataLoader):
    def can_handle(self, path: str) -> bool:
        return path.endswith(".sqlite") or path.endswith(".db")

    def load(self, path: str) -> list[dict]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            ]
            records = []
            for table in tables:
                rows = conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                for row in rows:
                    record = dict(row)
                    record["_table"] = table
                    records.append(record)
            return records
        finally:
            conn.close()
