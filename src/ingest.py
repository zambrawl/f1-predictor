"""Scarica/aggiorna i dati nel database SQLite (storico + stagione corrente)."""
import time
import pandas as pd
from . import db, jolpica, weather

RES_COLS = ["season", "round", "race_name", "date", "time", "lat", "lon", "circuit_id",
            "driver_id", "code", "team_id", "grid", "finish", "status", "points"]


def refresh(seasons):
    for s in seasons:
        r = pd.DataFrame(jolpica.rows(f"{s}/results", "Results")).rename(columns={"pos": "finish"})
        if len(r):
            db.upsert("results", r[RES_COLS], ["season", "round", "driver_id"])
        q = pd.DataFrame(jolpica.rows(f"{s}/qualifying", "QualifyingResults")).rename(columns={"pos": "qpos"})
        if len(q):
            db.upsert("quali", q[["season", "round", "driver_id", "qpos"]], ["season", "round", "driver_id"])
        if s >= 2021:   # le sprint esistono dal 2021
            sp = pd.DataFrame(jolpica.rows(f"{s}/sprint", "SprintResults")).rename(columns={"pos": "sprint_pos"})
            if len(sp):
                db.upsert("sprint", sp[["season", "round", "driver_id", "sprint_pos"]],
                          ["season", "round", "driver_id"])
        print(f"[ingest] stagione {s} ok")
    fill_weather()


def fill_weather():
    res, w = db.read("results"), db.read("weather")
    races = res.drop_duplicates(["season", "round"])[["season", "round", "date", "time", "lat", "lon"]]
    if len(w):
        done = w[w["rain_mm"].notna()][["season", "round"]]
        races = races.merge(done, how="left", indicator=True)
        races = races[races["_merge"] == "left_only"].drop(columns="_merge")
    new = []
    for _, r in races.iterrows():
        rain, temp = weather.at(r["lat"], r["lon"], r["date"], r["time"])
        new.append({"season": int(r["season"]), "round": int(r["round"]), "rain_mm": rain, "temp": temp})
        time.sleep(0.25)
    if new:
        db.upsert("weather", pd.DataFrame(new), ["season", "round"])
        print(f"[ingest] meteo aggiornato per {len(new)} gare")
