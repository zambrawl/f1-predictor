"""Costruzione delle feature. Ogni feature usa SOLO informazioni precedenti alla gara (shift(1)),
cosi' il modello non "vede il futuro"."""
import numpy as np
import pandas as pd
from .config import HALF_LIFE_RACES

# stessa squadra che cambia nome nel tempo -> stessa "macchina" per lo storico
TEAM_LINEAGE = {"force_india": "aston_martin", "racing_point": "aston_martin",
                "toro_rosso": "rb", "alphatauri": "rb",
                "lotus_f1": "alpine", "renault": "alpine",
                "alfa": "sauber", "audi": "sauber", "marussia": "manor"}
TEAM_NAMES = {"red_bull": "Red Bull", "ferrari": "Ferrari", "mercedes": "Mercedes", "mclaren": "McLaren",
              "aston_martin": "Aston Martin", "alpine": "Alpine", "williams": "Williams", "rb": "Racing Bulls",
              "sauber": "Sauber", "haas": "Haas", "manor": "Manor", "cadillac": "Cadillac", "caterham": "Caterham"}


def team_name(tid):
    tid = TEAM_LINEAGE.get(tid, tid)
    return TEAM_NAMES.get(tid, str(tid).replace("_", " ").title())


PRE = ["drv_pos5", "drv_win5", "drv_pts5", "drv_dnf10", "drv_ytd_pos", "drv_mate_delta5", "exp_races",
       "team_pos5", "team_pts5", "team_dnf10", "team_ytd_pos", "team_qgap5", "drv_qgap5",
       "drv_wet_gain", "rain_mm", "drv_circ_pos", "fp_best_gap", "fp_long_gap"]
POST = PRE + ["grid", "qgap", "sprint_pos", "ahead_pos5", "behind_pos5", "circ_pole_win"]


def _secs(t):
    try:
        m, s = str(t).split(":")
        return int(m) * 60 + float(s)
    except Exception:
        try:
            return float(t)
        except Exception:
            return np.nan


def _roll(df, by, col, n):
    return df.groupby(by)[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())


def _empty_sprint():
    return pd.DataFrame({"season": pd.Series(dtype="int64"), "round": pd.Series(dtype="int64"),
                         "driver_id": pd.Series(dtype="object"), "sprint_pos": pd.Series(dtype="float64")})


def make_features(res, quali, sprint, weather, practice=None):
    df = res.copy()
    df["team_id"] = df["team_id"].replace(TEAM_LINEAGE)
    df = df.sort_values(["season", "round"], kind="stable").reset_index(drop=True)
    keys = ["season", "round", "driver_id"]

    q = quali.copy()
    if "q1" not in q:
        q["q1"] = np.nan
    q["q1s"] = q["q1"].map(_secs)       # distacco % dal piu' veloce in Q1 (tutti hanno un tempo in Q1)
    q["qgap"] = (q["q1s"] / q.groupby(["season", "round"])["q1s"].transform("min") - 1) * 100
    df = df.merge(q[keys + ["qpos", "qgap"]], on=keys, how="left")
    if sprint is None or sprint.empty:
        sprint = _empty_sprint()
    df = df.merge(sprint[keys + ["sprint_pos"]], on=keys, how="left")
    df = df.merge(weather[["season", "round", "rain_mm"]], on=["season", "round"], how="left")
    if practice is not None and len(practice):
        df = df.merge(practice[["season", "round", "code", "fp_best_gap", "fp_long_gap"]],
                      on=["season", "round", "code"], how="left")
    else:
        df["fp_best_gap"] = np.nan
        df["fp_long_gap"] = np.nan

    mx = df.groupby(["season", "round"])["grid"].transform("max")
    df["grid"] = df["grid"].where(df["grid"] != 0, mx + 1)      # partenza dai box = dopo l'ultimo
    df["dnf"] = (~df["status"].fillna("").str.startswith(("Finished", "+"))).astype(float)
    df.loc[df["finish"].isna(), "dnf"] = np.nan
    df["win"] = np.where(df["finish"].isna(), np.nan, (df["finish"] == 1).astype(float))
    df["gain"] = df["grid"] - df["finish"]
    g = df.groupby(["season", "round", "team_id"])["finish"]
    n_t, s_t = g.transform("count"), g.transform("sum")
    df["mate_delta"] = df["finish"] - (s_t - df["finish"]) / (n_t - 1).replace(0, np.nan)  # pilota vs compagno

    # --- forma del pilota
    d = "driver_id"
    df["raw_pos"] = _roll(df, d, "finish", 5)
    df["raw_win"] = _roll(df, d, "win", 5)
    df["raw_pts"] = _roll(df, d, "points", 5)
    df["raw_dnf"] = _roll(df, d, "dnf", 10)
    df["raw_mate"] = _roll(df, d, "mate_delta", 10)
    df["raw_qgap"] = _roll(df, d, "qgap", 5)
    df["raw_ytd"] = df.groupby([d, "season"])["finish"].transform(lambda s: s.shift(1).expanding().mean())

    # --- forza della macchina (media del team, ultime gare e stagione in corso)
    t = (df.groupby(["team_id", "season", "round"], as_index=False)
           .agg(t_pos=("finish", "mean"), t_pts=("points", "sum"), t_dnf=("dnf", "mean"), t_qgap=("qgap", "mean"))
           .sort_values(["season", "round"]))
    for src, dst, n in (("t_pos", "team_pos5", 5), ("t_pts", "team_pts5", 5),
                        ("t_dnf", "team_dnf10", 10), ("t_qgap", "team_qgap5", 5)):
        t[dst] = t.groupby("team_id")[src].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
    t["team_ytd_pos"] = t.groupby(["team_id", "season"])["t_pos"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(t[["team_id", "season", "round", "team_pos5", "team_pts5", "team_dnf10",
                     "team_qgap5", "team_ytd_pos"]], on=["team_id", "season", "round"], how="left")
    df["team_pos5"] = df["team_pos5"].fillna(12)
    df["team_pts5"] = df["team_pts5"].fillna(0)
    df["team_dnf10"] = df["team_dnf10"].fillna(0.15)
    df["team_ytd_pos"] = df["team_ytd_pos"].fillna(df["team_pos5"])

    # --- esperienza e piloti nuovi: si parte dalla macchina e si passa gradualmente al pilota
    df["exp_races"] = df.groupby(d).cumcount()
    w = df["exp_races"].clip(upper=5) / 5
    df["drv_pos5"] = w * df["raw_pos"].fillna(0) + (1 - w) * df["team_pos5"]
    df["drv_win5"] = w * df["raw_win"].fillna(0)
    df["drv_pts5"] = w * df["raw_pts"].fillna(0) + (1 - w) * df["team_pts5"] / 2
    df["drv_dnf10"] = w * df["raw_dnf"].fillna(0) + (1 - w) * df["team_dnf10"]
    df["drv_mate_delta5"] = w * df["raw_mate"].fillna(0)
    df["drv_ytd_pos"] = df["raw_ytd"].fillna(df["drv_pos5"])
    df["drv_qgap5"] = w * df["raw_qgap"].fillna(df["team_qgap5"]) + (1 - w) * df["team_qgap5"]

    # --- storico del pilota su QUESTA pista (media posizioni, con prudenza se poche gare)
    cg = df.groupby([d, "circuit_id"])["finish"]
    c_sum = cg.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    c_n = cg.transform(lambda s: s.shift(1).expanding().count()).fillna(0)
    df["drv_circ_pos"] = (c_sum + 2 * df["drv_pos5"]) / (c_n + 2)

    # --- rendimento sul bagnato (posizioni guadagnate in gare con pioggia, con prudenza)
    df["gain_wet"] = df["gain"].where(df["rain_mm"].fillna(0) > 1)
    wg = df.groupby(d)["gain_wet"]
    w_sum = wg.transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    w_n = wg.transform(lambda s: s.shift(1).expanding().count()).fillna(0)
    df["drv_wet_gain"] = w_sum / (w_n + 3)

    # --- quanto conta la pole su questo circuito
    df["_pw"] = ((df["finish"] == 1) & (df["grid"] == 1)).astype(float)
    cp = (df.groupby(["circuit_id", "season", "round"], as_index=False)["_pw"].max()
            .sort_values(["season", "round"]))
    cp["circ_pole_win"] = cp.groupby("circuit_id")["_pw"].transform(lambda s: s.shift(1).expanding().mean())
    df = df.merge(cp[["circuit_id", "season", "round", "circ_pole_win"]],
                  on=["circuit_id", "season", "round"], how="left")
    df["circ_pole_win"] = df["circ_pole_win"].fillna(0.4)

    # --- chi parte davanti e dietro (in forma)
    df = df.sort_values(["season", "round", "grid"])
    gg = df.groupby(["season", "round"])["drv_pos5"]
    df["ahead_pos5"] = gg.shift(1)
    df["behind_pos5"] = gg.shift(-1)

    # --- peso dei campioni: gare recenti contano di piu'
    idx = (df["season"] * 100 + df["round"]).rank(method="dense")
    df["w"] = 0.5 ** ((idx.max() - idx) / HALF_LIFE_RACES)
    return df.sort_values(["season", "round", "grid"]).reset_index(drop=True)
