"""Comandi Telegram, penalita', orari delle notifiche (solo ven-dom), riaddestramento. Rete simulata."""
import datetime as dt
import os
import pandas as pd
from tests.synth import make
from src import db, grid, ingest, jolpica, model, notify, practice, run, weather
from src.features import make_features
from src.config import DB, DATA

assert grid.apply_penalties(["A", "B", "C", "D"], [("A", 2)]) == {"B": 1, "C": 2, "A": 3, "D": 4}
assert grid.apply_penalties(["A", "B", "C", "D"], [("B", 99)]) == {"A": 1, "C": 2, "D": 3, "B": 4}

if DB.exists():
    DB.unlink()
res, q, sp, w, fp = make(rounds=8)
for n, d in (("results", res), ("quali", q), ("sprint", sp), ("weather", w)):
    db.write(n, d)
df = make_features(res, q, sp, w, practice.aggregate(fp))
model.train_all(df); model.mark_trained(df)

# prossimo GP: round 9 del 2026, sabato 16 maggio 2026 qualifiche 14:00 UTC, gara domenica 17
UTC = dt.timezone.utc
qr9 = pd.DataFrame([dict(season=2026, round=9, driver_id=f"d{i}", code=f"D{i:02d}", team_id="x", qpos=i + 1,
                         q1=f"1:20.{i:03d}", q2=f"1:19.{i:03d}" if i < 15 else None,
                         q3=f"1:18.{i:03d}" if i < 10 else None) for i in range(20)])
# team reale per le righe della qualifica
qr9["team_id"] = [f"{t}" for t in (["red_bull", "ferrari", "mercedes", "mclaren", "aston_martin", "alpine",
                                    "williams", "rb", "sauber", "haas"] * 2)][:20]
nxt = dict(season=2026, round=9, race_name="GP9", date="2026-05-17", time="13:00:00Z", lat=0.0, lon=0.0,
           circuit_id="c3", sessions=[("Prove libere 1", dt.datetime(2026, 5, 15, 11, 30, tzinfo=UTC), "FP1"),
                                      ("Qualifiche", dt.datetime(2026, 5, 16, 14, 0, tzinfo=UTC), "Q"),
                                      ("Gara", dt.datetime(2026, 5, 17, 13, 0, tzinfo=UTC), "R")])
jolpica.next_race = lambda: nxt
ingest.refresh_smart = lambda *a, **k: None
weather.forecast = lambda n: (0.0, 20.0)
grid._fastf1_grid = lambda s, r: None
practice.update_current = lambda *a, **k: None
msgs, updates = [], []
notify.send = lambda t: msgs.append(t)
notify.get_updates = lambda offset=None: [u for u in updates if offset is None or u["update_id"] >= offset]
os.environ["TELEGRAM_CHAT_ID"] = "42"


def at(y, m, d, h, mi):
    run.NOW = dt.datetime(y, m, d, h, mi, tzinfo=UTC)


def say(uid, text, chat=42):
    updates.append({"update_id": uid, "message": {"chat": {"id": chat}, "text": text}})


# --- 1) giovedi': nessuna notifica automatica nemmeno 30 min prima (non e' un giorno di gara)
at(2026, 5, 14, 12, 0); run.main(); assert msgs == [], msgs
# --- 2) venerdi' 11:05 UTC: promemoria FP1 (25 minuti prima), una sola volta
at(2026, 5, 15, 11, 5); run.main(); assert len(msgs) == 1 and "Prove libere 1" in msgs[0] and "25 minuti" in msgs[0]
assert "1." in msgs[0] and "10." in msgs[0]
run.main(); assert len(msgs) == 1, "doppio invio"
# --- 3) comandi: /classifica non esiste piu', /classificagara in fase 'pre' con punteggi e macchine
n = len(msgs)
say(1, "/classifica"); say(2, "/classificagara"); say(3, "/classificaqualifiche"); say(4, "/aiuto")
at(2026, 5, 15, 12, 0); run.main()
out = msgs[n:]
assert "Comandi:" in out[0]                                   # /classifica -> elenco comandi
assert "Pronostico gara" in out[1] and "Macchina" in out[1] and "Pioggia" in out[1] and "Peso dei fattori" in out[1]
assert "Red Bull" in out[1] and "10." in out[1]
assert "non ancora disputate" in out[2]                       # nessuna qualifica, nessuna libera
print(out[1])
# --- 4) comandi funzionano anche il lunedi' (risposta a richiesta), notifiche automatiche no
n = len(msgs); say(5, "/classificagara"); at(2026, 5, 11, 10, 0); run.main(); assert len(msgs) == n + 1
# --- 5) qualifiche disputate: report automatico + tempi con /classificaqualifiche
db.upsert("quali", qr9, ["season", "round", "driver_id"])
n = len(msgs); at(2026, 5, 16, 15, 0); run.main()
assert "dopo le qualifiche" in msgs[n] and "ordine di qualifica" in msgs[n]
base_top = msgs[n].split("Top 10")[1]
say(6, "/classificaqualifiche"); n = len(msgs); at(2026, 5, 16, 15, 5); run.main()
assert "Qualifiche – GP9" in msgs[n] and "(pole)" in msgs[n] and "(+0.0" in msgs[n] and "(Q2)" in msgs[n]
print(msgs[n])
# --- 6) penalita' da telefono
for uid, t in ((7, "/penalita VER 3"), (8, "/penalita d0 3"), (9, "/penalità d1 box"), (10, "/penalita lista"),
               (11, "/penalita pippo 2"), (12, "/penalita d1 abc")):
    say(uid, t)
n = len(msgs); at(2026, 5, 16, 15, 10); run.main()
o = msgs[n:]
assert "non trovato" in o[0]                                  # 'VER' non esiste nei dati finti
assert "D00 -3 posti" in o[1]
assert "D01 parte dai box" in o[2]
assert "D00 -3 posti, D01 dai box/ultimo" in o[3]
assert "non trovato" in o[4] and "posti persi" in o[5]
g = grid.resolve(2026, 9, qr9.rename(columns={"qpos": "grid"}))
assert g[0][qr9.code == "D00"].iloc[0] == 3 and g[0][qr9.code == "D01"].iloc[0] == 20, g[0].tolist()
say(13, "/classificagara"); n = len(msgs); at(2026, 5, 16, 15, 15); run.main()
assert "Penalita applicate: D00 -3, D01 dai box" in msgs[n] and "penalita' inserite da te" in msgs[n]
say(14, "/penalita cancella D01"); say(15, "/penalita reset"); n = len(msgs); at(2026, 5, 16, 15, 20); run.main()
assert "rimossa" in msgs[n] and "azzerate" in msgs[n + 1] and grid.get_penalties(2026, 9) == []
# --- 7) comandi da chat estranee ignorati
n = len(msgs); say(16, "/classificagara", chat=999); at(2026, 5, 16, 15, 25); run.main(); assert len(msgs) == n
# --- 8) promemoria gara (domenica 12:35 UTC) e qualifiche gia' mandati una volta
n = len(msgs); at(2026, 5, 17, 12, 35); run.main()
assert any("Gara – GP9" in m and "25 minuti" in m for m in msgs[n:])
# --- 9) gara finita: /classificagara mostra i tempi, notifica esito domenica, riaddestramento
r9 = pd.DataFrame([dict(season=2026, round=9, race_name="GP9", date="2026-05-17", time="13:00:00Z", lat=0.0, lon=0.0,
                        circuit_id="c3", driver_id=f"d{i}", code=f"D{i:02d}", team_id=qr9.team_id[i], grid=i + 1,
                        finish=i + 1, status="Finished", points=0.0,
                        race_time="1:31:11.111" if i == 0 else f"+{i * 2.5:.1f}s") for i in range(20)])
db.upsert("results", r9, ["season", "round", "driver_id"])
db.upsert("weather", pd.DataFrame([dict(season=2026, round=9, rain_mm=0.0, temp=20.0)]), ["season", "round"])
say(17, "/classificagara"); n = len(msgs); at(2026, 5, 17, 19, 0); run.main()
assert any("Ha vinto D00" in m or "ha vinto D00" in m for m in msgs[n:]), msgs[n:]
assert any("Risultato gara" in m and "1:31:11.111" in m and "+2.5s" in m for m in msgs[n:])
assert int(db.read("meta")["v"].iloc[0]) == 2026 * 100 + 9
print("LOGIC TEST OK")
