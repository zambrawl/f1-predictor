"""Costruzione delle feature. Ogni feature usa SOLO informazioni precedenti alla gara (shift(1)),
cosi' il modello non "vede il futuro"."""
import numpy as np
import pandas as pd
from .config import HALF_LIFE_RACES

PRE = ["drv_pos5", "drv_win5", "drv_pts5", "drv_dnf10",
       "team_pos5", "team_pts5", "team_dnf10", "drv_wet_gain", "rain_mm"]
POST = PRE + ["grid", "sprint_pos", "ahead_pos5", "behind_pos5", "circ_pole_win"]


def _roll(df, by, col, n):
    return df.groupby(by)[col].transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean())


def make_features(res, quali, sprint, weather):
    df = res.sort_values(["season", "round"]).reset_index(drop=True)
    keys = ["season", "round", "driver_id"]
    df = df.merge(quali[keys + ["qpos"]], on=keys, how="left")
    df = df.merge(sprint[keys + ["sprint_pos"]], on=keys, how="left")
    df = df.merge(weather[["season", "round", "rain_mm"]], on=["season", "round"], how="left")

    df["grid"] = df["grid"].replace(0, 20)                      # partenza dai box = ultimo
    df["dnf"] = (~df["status"].fillna("").str.startswith(("Finished", "+"))).astype(float)
    df.loc[df["finish"].isna(), "dnf"] = np.nan
    df["win"] = np.where(df["finish"].isna(), np.nan, (df["finish"] == 1).astype(float))
    df["gain"] = df["grid"] - df["finish"]

    # --- forma del pilota (ultime gare)
    d = "driver_id"
    df["raw_pos"] = _roll(df, d, "finish", 5)
    df["raw_win"] = _roll(df, d, "win", 5)
    df["raw_pts"] = _roll(df, d, "points", 5)
    df["raw_dnf"] = _roll(df, d, "dnf", 10)

    # --- forza della macchina (media del team nelle ultime gare)
    t = (df.groupby(["team_id", "season", "round"], as_index=False)
           .agg(t_pos=("finish", "mean"), t_pts=("points", "sum"), t_dnf=("dnf", "mean"))
           .sort_values(["season", "round"]))
    for src, dst, n in (("t_pos", "team_pos5", 5), ("t_pts", "team_pts5", 5), ("t_dnf", "team_dnf10", 10)):
        t[dst] = t.groupby("team_id")[src].transform(lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
    df = df.merge(t[["team_id", "season", "round", "team_pos5", "team_pts5", "team_dnf10"]],
                  on=["team_id", "season", "round"], how="left")
    df["team_pos5"] = df["team_pos5"].fillna(12)
    df["team_pts5"] = df["team_pts5"].fillna(0)
    df["team_dnf10"] = df["team_dnf10"].fillna(0.15)

    # --- piloti nuovi / pochi dati: si parte dalla macchina e si passa gradualmente al pilota
    w = (df.groupby(d).cumcount().clip(upper=5) / 5)
    df["drv_pos5"] = w * df["raw_pos"].fillna(0) + (1 - w) * df["team_pos5"]
    df["drv_win5"] = w * df["raw_win"].fillna(0)
    df["drv_pts5"] = w * df["raw_pts"].fillna(0) + (1 - w) * df["team_pts5"] / 2
    df["drv_dnf10"] = w * df["raw_dnf"].fillna(0) + (1 - w) * df["team_dnf10"]

    # --- rendimento storico sul bagnato (posizioni guadagnate in gare con pioggia)
    wet = df["rain_mm"].fillna(0) > 1
    df["gain_wet"] = df["gain"].where(wet)
    df["drv_wet_gain"] = (df.groupby(d)["gain_wet"]
                            .transform(lambda s: s.shift(1).expanding().mean()).fillna(0))

    # --- quanto conta la pole su questo circuito
    df["_pw"] = ((df["finish"] == 1) & (df["grid"] == 1)).astype(float)
    cp = (df.groupby(["circuit_id", "season", "round"], as_index=False)["_pw"].max()
            .sort_values(["season", "round"]))
    cp["circ_pole_win"] = cp.groupby("circuit_id")["_pw"].transform(
        lambda s: s.shift(1).expanding().mean())
    df = df.merge(cp[["circuit_id", "season", "round", "circ_pole_win"]],
                  on=["circuit_id", "season", "round"], how="left")
    df["circ_pole_win"] = df["circ_pole_win"].fillna(0.4)

    # --- chi parte davanti e dietro (in forma)
    df = df.sort_values(["season", "round", "grid"])
    g = df.groupby(["season", "round"])["drv_pos5"]
    df["ahead_pos5"] = g.shift(1)
    df["behind_pos5"] = g.shift(-1)

    # --- peso dei campioni: gare recenti contano di piu'
    idx = (df["season"] * 100 + df["round"]).rank(method="dense")
    df["w"] = 0.5 ** ((idx.max() - idx) / HALF_LIFE_RACES)
    return df.sort_values(["season", "round", "grid"]).reset_index(drop=True)
