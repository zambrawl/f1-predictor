"""Scarica/aggiorna i dati nel database SQLite."""
import datetime as dt
import time
import pandas as pd
from . import db, jolpica, weather

KEYS = ["season", "round", "driver_id"]
RES_COLS = ["season", "round", "race_name", "date", "time", "lat", "lon", "circuit_id", "driver_id",
            "code", "team_id", "grid", "finish", "status", "points", "race_time"]
Q_COLS = ["season", "round", "driver_id", "code", "team_id", "qpos", "q1", "q2", "q3"]


def _store(res_rows, q_rows, sp_rows):
    r = pd.DataFrame(res_rows)
    if len(r):
        db.upsert("results", r.rename(columns={"pos": "finish"})[RES_COLS], KEYS)
    q = pd.DataFrame(q_rows)
    if len(q):
        db.upsert("quali", q.rename(columns={"pos": "qpos"})[Q_COLS], KEYS)
    s = pd.DataFrame(sp_rows)
    if len(s):
        db.upsert("sprint", s.rename(columns={"pos": "sprint_pos"})[KEYS + ["sprint_pos"]], KEYS)


def _sprint(path, season):
    return jolpica.rows(path, "SprintResults") if season >= 2021 else []   # le sprint esistono dal 2021


def refresh(seasons):
    """Aggiornamento completo di intere stagioni (usato da bootstrap)."""
    for s in seasons:
        _store(jolpica.rows(f"{s}/results", "Results"),
               jolpica.rows(f"{s}/qualifying", "QualifyingResults"),
               _sprint(f"{s}/sprint", s))
        print(f"[ingest] stagione {s} ok")
    fill_weather()


def refresh_recent(season, rounds):
    for r in rounds:
        _store(jolpica.rows(f"{season}/{r}/results", "Results"),
               jolpica.rows(f"{season}/{r}/qualifying", "QualifyingResults"),
               _sprint(f"{season}/{r}/sprint", season))
    fill_weather()


def refresh_smart(nxt, now, full_ok=True):
    """Pochissime chiamate per esecuzione: solo ultima gara e gara in corso.
    Se mancano gare precedenti nella stagione fa (al massimo una volta al giorno) un aggiornamento completo."""
    if nxt is None:
        if now.minute < 10:          # fuori stagione: una volta all'ora
            refresh([now.year])
            return "full"
        return None
    s, r = nxt["season"], nxt["round"]
    res = db.read("results")
    have = set(res[res["season"] == s]["round"]) if len(res) else set()
    missing = [x for x in range(1, r) if x not in have]
    if missing and full_ok:
        refresh([s])
        return "full"
    refresh_recent(s, [x for x in (r - 1, r) if x >= 1])
    return "recent"


def fill_weather(today=None):
    today = today or dt.date.today()
    res, w = db.read("results"), db.read("weather")
    races = res.drop_duplicates(["season", "round"])[["season", "round", "date", "time", "lat", "lon"]]
    if len(w):
        m = races.merge(w[["season", "round", "rain_mm"]], how="left", on=["season", "round"], indicator=True)
        recent = pd.to_datetime(m["date"], errors="coerce") >= pd.Timestamp(today) - pd.Timedelta(days=14)
        races = m[(m["_merge"] == "left_only") | (m["rain_mm"].isna() & recent)]
    new = []
    for _, r in races.iterrows():
        rain, temp = weather.at(r["lat"], r["lon"], r["date"], r["time"])
        new.append({"season": int(r["season"]), "round": int(r["round"]), "rain_mm": rain, "temp": temp})
        time.sleep(0.25)
    if new:
        db.upsert("weather", pd.DataFrame(new), ["season", "round"])
        print(f"[ingest] meteo aggiornato per {len(new)} gare")
