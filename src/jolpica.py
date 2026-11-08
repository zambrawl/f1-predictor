"""Client per Jolpica-F1 (successore di Ergast): calendario, risultati, qualifiche, sprint."""
import time
import requests

BASE = "https://api.jolpi.ca/ergast/f1"


def _get(path, params=None):
    for i in range(6):
        r = requests.get(f"{BASE}/{path}.json", params=params, timeout=30)
        if r.status_code == 429:          # troppe richieste: aspetto e riprovo
            time.sleep(5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()["MRData"]
    raise RuntimeError("Jolpica: troppe richieste, riprova piu' tardi")


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _meta(race):
    loc = race["Circuit"]["Location"]
    return {
        "season": int(race["season"]), "round": int(race["round"]),
        "race_name": race["raceName"], "date": race["date"],
        "time": race.get("time", "13:00:00Z"),
        "lat": float(loc["lat"]), "lon": float(loc["long"]),
        "circuit_id": race["Circuit"]["circuitId"],
    }


def rows(path, inner):
    """Scarica tutte le righe (con paginazione). inner = Results | QualifyingResults | SprintResults"""
    out, offset = [], 0
    while True:
        m = _get(path, {"limit": 100, "offset": offset})
        for race in m["RaceTable"]["Races"]:
            meta = _meta(race)
            for r in race.get(inner, []):
                d = r["Driver"]
                out.append({
                    **meta,
                    "driver_id": d["driverId"],
                    "code": d.get("code", d["driverId"][:3].upper()),
                    "team_id": r["Constructor"]["constructorId"],
                    "pos": _int(r.get("position")),
                    "grid": _int(r.get("grid")),
                    "status": r.get("status"),
                    "points": float(r.get("points", 0) or 0),
                })
        offset += 100
        if offset >= int(m["total"]):
            break
        time.sleep(0.3)
    return out


def next_race():
    races = _get("current/next")["RaceTable"]["Races"]
    return _meta(races[0]) if races else None
