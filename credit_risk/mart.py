"""SQLite analogue of a point-in-time feature-store snapshot."""
from pathlib import Path
import sqlite3
import pandas as pd


def build_mart(data: pd.DataFrame, db_path: str | Path) -> pd.DataFrame:
    db = Path(db_path); db.parent.mkdir(parents=True, exist_ok=True)
    if data.application_id.duplicated().any():
        raise ValueError('duplicate application_id')
    if not set(data.default_12m.dropna().unique()).issubset({0,1}) or data.default_12m.isna().any():
        raise ValueError('default_12m must be 0 or 1')
    if data.application_date.isna().any():
        raise ValueError('missing application_date')
    with sqlite3.connect(db) as connection:
        data.to_sql('raw_applications', connection, if_exists='replace', index=False)
        script = (Path(__file__).parents[1] / 'sql' / '01_feature_mart.sql').read_text(encoding='utf-8')
        connection.executescript(script)
        mart = pd.read_sql_query('SELECT * FROM feature_store_applications ORDER BY application_date, application_id', connection)
    if len(data) != len(mart):
        raise AssertionError('lost rows during feature mart creation')
    if abs(float(data.default_12m.sum()) - float(mart.default_12m.sum())) > 1e-6:
        raise AssertionError('target reconciliation failed')
    return mart
