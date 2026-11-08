import sqlite3
import pandas as pd
from .config import DB


def _conn():
    return sqlite3.connect(DB)


def read(table):
    with _conn() as c:
        try:
            return pd.read_sql(f"select * from {table}", c)
        except Exception:
            return pd.DataFrame()


def write(table, df):
    with _conn() as c:
        df.to_sql(table, c, if_exists="replace", index=False)


def upsert(table, new, keys):
    """Aggiunge/aggiorna righe. Non riscrive il file se nulla e' cambiato
    (cosi' GitHub non crea commit inutili)."""
    old = read(table)
    df = new if old.empty else pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    if not old.empty:
        if set(old.columns) == set(df.columns):
            df = df[old.columns]
        o = old.sort_values(keys).reset_index(drop=True)
        try:
            if o.equals(df):
                return df
        except Exception:
            pass
    write(table, df)
    return df
