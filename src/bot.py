"""Comandi Telegram: parsing e gestione delle penalita' in griglia."""
import unicodedata
import pandas as pd
from . import db

HELP = ("Comandi:\n"
        "/classificagara – pronostico gara con percentuali, macchine e punteggi (dopo la gara: tempi ufficiali)\n"
        "/classificaqualifiche – tempi delle qualifiche\n"
        "/penalita – gestisci le penalita' in griglia\n"
        "/aiuto – questo elenco\n\n"
        "Penalita':\n"
        "/penalita VER 3  → VER perde 3 posti\n"
        "/penalita HAM box  → parte dai box / ultimo\n"
        "/penalita lista\n"
        "/penalita cancella VER\n"
        "/penalita reset")
BOX = {"box", "pitlane", "pit", "ultimo", "fondo", "last"}


def _norm(t):
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()


def parse(text):
    parts = _norm(text.strip()).split()
    return parts[0].split("@")[0], parts[1:]


def _active(season, rnd):
    p = db.read("penalties")
    if p.empty:
        return pd.DataFrame(columns=["season", "round", "code", "places"])
    return p[(p["season"] == season) & (p["round"] == rnd)]


def _fmt(p):
    if p.empty:
        return "nessuna"
    return ", ".join(f"{r.code} " + ("dai box/ultimo" if r.places >= 99 else f"-{int(r.places)} posti")
                     for r in p.itertuples())


def _resolve(tok, lineup):
    t = _norm(tok)
    for r in lineup.itertuples():
        if t == r.code.lower() or t in r.driver_id.lower().split("_"):
            return r.code
    return None


def _remove(season, rnd, code=None):
    p = db.read("penalties")
    if p.empty:
        return
    mask = (p["season"] == season) & (p["round"] == rnd)
    if code:
        mask &= p["code"] == code
    db.write("penalties", p[~mask])


def penalty_command(args, nxt, lineup):
    if not nxt:
        return "Nessuna gara in programma: non posso registrare penalita'."
    s, r, name = nxt["season"], nxt["round"], nxt["race_name"]
    if not args or args[0] in ("lista", "list"):
        return f"Penalita' attive per {name}: {_fmt(_active(s, r))}\n\nUso: /penalita VER 3 · /penalita HAM box · /penalita cancella VER · /penalita reset"
    if args[0] == "reset":
        _remove(s, r)
        return f"✅ Penalita' azzerate per {name}."
    if args[0] in ("cancella", "rimuovi", "togli") and len(args) > 1:
        code = _resolve(args[1], lineup)
        if not code:
            return f"Pilota '{args[1]}' non trovato. Usa la sigla (es. VER) o il cognome."
        _remove(s, r, code)
        return f"✅ Penalita' di {code} rimossa. Attive: {_fmt(_active(s, r))}"
    if len(args) >= 2:
        code = _resolve(args[0], lineup)
        if not code:
            return f"Pilota '{args[0]}' non trovato. Usa la sigla (es. VER) o il cognome."
        if args[1] in BOX:
            places = 99
        else:
            try:
                places = int(args[1])
            except ValueError:
                return "Dopo il pilota scrivi i posti persi (es. 3) oppure 'box'."
            if places <= 0:
                return "I posti persi devono essere un numero positivo."
        db.upsert("penalties", pd.DataFrame([{"season": s, "round": r, "code": code, "places": places}]),
                  ["season", "round", "code"])
        return (f"✅ Penalita' registrata per {name}: {code} " + ("parte dai box/ultimo" if places >= 99 else f"-{places} posti")
                + f"\nAttive: {_fmt(_active(s, r))}\nLa griglia nelle previsioni e' corretta da subito: scrivi /classificagara.")
    return "Non ho capito. Esempi: /penalita VER 3 · /penalita HAM box · /penalita lista"
