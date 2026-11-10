"""Griglia di partenza corretta. Fonti, in ordine di priorita':
1) penalita' inserite da te dal bot (/penalita VER 3) -> applicate all'ordine di qualifica
2) FastF1: GridPosition ufficiale, se disponibile e valida
3) ordine di qualifica (senza penalita')
"""
import pandas as pd
from . import db


def apply_penalties(order, penalties):
    order = list(order)
    # si applicano partendo da chi si era qualificato meglio (non in ordine alfabetico)
    pos = {c: i for i, c in enumerate(order)}
    for code, places in sorted(penalties, key=lambda p: pos.get(p[0], 99)):
        if code not in order:
            continue
        i = order.index(code)
        order.pop(i)
        j = len(order) if places >= 99 else min(i + int(places), len(order))
        order.insert(j, code)
    return {c: i + 1 for i, c in enumerate(order)}


def get_penalties(season, rnd):
    p = db.read("penalties")
    if p.empty:
        return []
    p = p[(p["season"] == season) & (p["round"] == rnd)]
    return list(zip(p["code"], p["places"]))


def _fastf1_grid(season, rnd):
    try:
        import tempfile
        import fastf1
        fastf1.Cache.enable_cache(tempfile.mkdtemp())
        s = fastf1.get_session(season, rnd, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        r = s.results[["Abbreviation", "GridPosition"]].dropna()
        r = r[r["GridPosition"] >= 0]
        if len(r) >= 15 and r["GridPosition"].nunique() >= 10:     # dati sensati, non tutti zero
            return {a: int(p) for a, p in zip(r["Abbreviation"], r["GridPosition"])}
    except Exception:
        pass
    return None


def resolve(season, rnd, rows):
    """rows: DataFrame con 'code' e 'grid' (ordine di qualifica). Ritorna (griglia, fonte, penalita')."""
    pen = get_penalties(season, rnd)
    order = rows.sort_values("grid")["code"].tolist()
    if pen:
        return rows["code"].map(apply_penalties(order, pen)), "qualifiche + penalita' inserite da te", pen
    g = _fastf1_grid(season, rnd)
    if g and rows["code"].isin(g.keys()).all():
        return rows["code"].map(g), "FastF1 (griglia ufficiale)", []
    return rows["grid"], "ordine di qualifica (penalita' non incluse)", []
