"""Tempi delle prove libere con FastF1: miglior giro e passo sui long run (giri puliti: niente giri ai box,
niente safety car / VSC / bandiera rossa). Salvati in data/practice/<anno>_<round>.csv."""
import datetime as dt
import logging
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DATA

DIR = DATA / "practice"
FAILS = DATA / "practice_fails.csv"
KINDS = ("FP1", "FP2", "FP3")
COLS = ["season", "round", "session", "code", "best_s", "long_s"]


def _fastf1():
    import fastf1
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    p = Path(tempfile.gettempdir()) / "fastf1cache"
    p.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(p))
    return fastf1


def session_stats(season, rnd, kind):
    ff1 = _fastf1()
    s = ff1.get_session(season, rnd, kind)
    s.load(telemetry=False, weather=False, messages=False)
    laps = s.laps
    clean = laps[laps["LapTime"].notna() & laps["PitInTime"].isna() & laps["PitOutTime"].isna()
                 & (laps["TrackStatus"].astype(str) == "1")].copy()
    if clean.empty:
        return None
    clean["sec"] = clean["LapTime"].dt.total_seconds()
    best = clean.groupby("Driver")["sec"].min()
    clean = clean[clean["sec"] < best.min() * 1.07]
    st = clean.groupby(["Driver", "Stint"])["sec"].agg(["count", "median"]).reset_index()
    st = st[st["count"] >= 5]                                   # solo stint lunghi (>=5 giri puliti)
    long = st.groupby("Driver")["median"].min()
    out = pd.DataFrame({"best_s": best}).join(long.rename("long_s")).reset_index().rename(columns={"Driver": "code"})
    out["season"], out["round"], out["session"] = season, rnd, kind
    return out[COLS]


def _path(season, rnd):
    return DIR / f"{int(season)}_{int(rnd):02d}.csv"


def save(df):
    DIR.mkdir(parents=True, exist_ok=True)
    for (s, r), g in df.groupby(["season", "round"]):
        p = _path(s, r)
        old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLS)
        pd.concat([old, g[COLS]], ignore_index=True).drop_duplicates(["session", "code"], keep="last") \
            .to_csv(p, index=False)


def load_all():
    files = sorted(DIR.glob("*.csv")) if DIR.exists() else []
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=COLS)


def aggregate(sess):
    cols = ["season", "round", "code", "fp_best_gap", "fp_long_gap"]
    if sess is None or sess.empty:
        return pd.DataFrame(columns=cols)
    g = sess.groupby(["season", "round", "code"], as_index=False).agg(best_s=("best_s", "min"),
                                                                       long_s=("long_s", "min"))
    for src, dst in (("best_s", "fp_best_gap"), ("long_s", "fp_long_gap")):
        ref = g.groupby(["season", "round"])[src].transform("min")
        g[dst] = (g[src] / ref - 1) * 100                      # distacco % dal piu' veloce del weekend
    return g[cols]


def table():
    return aggregate(load_all())


def update_current(nxt, now, tried, mark):
    """Scarica le libere del weekend in corso appena finite (al massimo un tentativo all'ora per sessione)."""
    sess = load_all()
    have = set(zip(sess["season"], sess["round"], sess["session"]))
    s, r = nxt["season"], nxt["round"]
    for label, start, kind in nxt["sessions"]:
        if kind not in KINDS or (s, r, kind) in have:
            continue
        if not (start + dt.timedelta(minutes=75) <= now <= start + dt.timedelta(hours=36)):
            continue
        key = f"fptry-{s}-{r}-{kind}-{now:%Y%m%d%H}"
        if tried(key):
            continue
        mark(key)
        try:
            df = session_stats(s, r, kind)
            if df is not None and len(df):
                save(df)
                print(f"[libere] {kind} salvata")
        except Exception as e:
            print(f"[libere] {kind} non disponibile:", e)


def build_history(max_minutes=300, since=2018):
    """Scarica lo storico delle libere (FastF1 parte dal 2018). Riprendibile: salta cio' che c'e' gia'."""
    from . import db
    t0 = time.time()
    res = db.read("results")
    races = (res[res["season"] >= since].drop_duplicates(["season", "round"])[["season", "round"]]
             .sort_values(["season", "round"], ascending=False))
    have = load_all()
    done = set(zip(have["season"], have["round"], have["session"]))
    fails = pd.read_csv(FAILS) if FAILS.exists() else pd.DataFrame(columns=["season", "round", "session", "n"])
    nfail = {(a, b, c): n for a, b, c, n in fails[["season", "round", "session", "n"]].itertuples(index=False)}
    ok = 0
    for s, r in races.itertuples(index=False):
        for kind in KINDS:
            if (s, r, kind) in done or nfail.get((s, r, kind), 0) >= 2:
                continue
            if (time.time() - t0) / 60 > max_minutes:
                print("Tempo massimo raggiunto: rilancia per continuare.")
                _save_fails(nfail)
                return
            try:
                df = session_stats(int(s), int(r), kind)
            except Exception as e:
                df = None
                print(f"  {s} R{r} {kind}: non disponibile ({str(e)[:60]})")
            if df is not None and len(df):
                save(df)
                ok += 1
                print(f"  {s} R{r} {kind} ok ({ok})")
            else:
                nfail[(s, r, kind)] = nfail.get((s, r, kind), 0) + 1
        _save_fails(nfail)
    print("Storico libere completo.")


def _save_fails(nfail):
    pd.DataFrame([{"season": a, "round": b, "session": c, "n": n} for (a, b, c), n in nfail.items()],
                 columns=["season", "round", "session", "n"]).to_csv(FAILS, index=False)
