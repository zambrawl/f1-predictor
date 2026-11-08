"""Griglia di partenza corretta. Fonti, in ordine di priorita':
1) grid_override.csv  -> penalita' che inserisci tu (colonne: season,round,code,places;
                          places = posti persi, oppure 99 = partenza dai box / ultimo)
2) FastF1             -> GridPosition ufficiale, se disponibile e valida
3) Qualifiche Jolpica -> ordine di qualifica (senza penalita')
"""
import pandas as pd
from .config import ROOT


def apply_penalties(order, penalties):
    order = list(order)
    for code, places in penalties:
        if code not in order:
            continue
        i = order.index(code)
        order.pop(i)
        j = len(order) if places >= 99 else min(i + int(places), len(order))
        order.insert(j, code)
    return {c: i + 1 for i, c in enumerate(order)}


def _penalties(season, rnd):
    p = ROOT / "grid_override.csv"
    if not p.exists():
        return []
    o = pd.read_csv(p)
    o = o[(o["season"] == season) & (o["round"] == rnd)]
    return list(zip(o["code"], o["places"]))


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
    """rows: DataFrame con 'code' e 'grid' (ordine di qualifica). Ritorna (griglia, fonte)."""
    pen = _penalties(season, rnd)
    order = rows.sort_values("grid")["code"].tolist()
    if pen:
        return rows["code"].map(apply_penalties(order, pen)), "qualifiche + penalita' da grid_override.csv"
    g = _fastf1_grid(season, rnd)
    if g and rows["code"].isin(g.keys()).all():
        return rows["code"].map(g), "FastF1 (griglia ufficiale)"
    return rows["grid"], "ordine di qualifica (penalita' non incluse)"
