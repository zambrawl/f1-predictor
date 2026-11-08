import datetime as dt, numpy as np, pandas as pd
from src import db, model, run, grid, jolpica, ingest, weather, notify
from src.features import make_features
from src.config import DB

assert grid.apply_penalties(["A","B","C","D"], [("A", 2)]) == {"B":1,"C":2,"A":3,"D":4}
assert grid.apply_penalties(["A","B","C","D"], [("B", 99)]) == {"A":1,"C":2,"D":3,"B":4}

if DB.exists(): DB.unlink()
rng = np.random.default_rng(1)
drivers = [(f"d{i}", f"D{i:02d}", f"t{i//2}") for i in range(20)]
res, q, w = [], [], []
for s in (2025, 2026):
    for r in range(1, 6):
        st = {d: -(int(d[1:])//2)*1.2 + rng.normal(0, 1.5) for d,_,_ in drivers}
        order = sorted(st, key=st.get, reverse=True)
        for d, c, t in drivers:
            res.append(dict(season=s, round=r, race_name=f"GP{r}", date="x", time="13:00:00Z", lat=0, lon=0,
                            circuit_id=f"c{r}", driver_id=d, code=c, team_id=t, grid=order.index(d)+1,
                            finish=order.index(d)+1, status="Finished", points=0.0))
            q.append(dict(season=s, round=r, driver_id=d, qpos=order.index(d)+1))
        w.append(dict(season=s, round=r, rain_mm=0.0, temp=20.0))
res, q, w = map(pd.DataFrame, (res, q, w))
sp = pd.DataFrame(dict(season=[2025], round=[1], driver_id=["d0"], sprint_pos=[1]))
for n, d in (("results",res),("quali",q),("sprint",sp),("weather",w)): db.write(n, d)
model.train_all(make_features(res, q, sp, w))

# calendario simulato: GP6 con qualifiche gia' disputate (quali per round 6 nel db)
q6 = pd.DataFrame([dict(season=2026, round=6, driver_id=d, qpos=i+1) for i,(d,_,_) in enumerate(drivers)])
db.upsert("quali", q6, ["season","round","driver_id"])
now = dt.datetime(2026, 5, 16, 12, 40, tzinfo=dt.timezone.utc)
def mk(h, m=0): return dt.datetime(2026, 5, 16, h, m, tzinfo=dt.timezone.utc)
nxt = dict(season=2026, round=6, race_name="GP6", date="2026-05-17", time="13:00:00Z", lat=0, lon=0, circuit_id="c6",
           sessions=[("Qualifiche", mk(13)), ("Gara", dt.datetime(2026,5,17,13,tzinfo=dt.timezone.utc))])
run.NOW = now
jolpica.next_race = lambda: nxt
ingest.refresh = lambda seasons: None
weather.forecast = lambda n: (0.0, 20.0)
grid._fastf1_grid = lambda s, r: None
msgs = []
notify.send = lambda t: msgs.append(t)
run.main()
print("\n\n=====\n".join(msgs))
assert any("Tra circa 20 minuti" in m for m in msgs) and any("dopo le qualifiche" in m for m in msgs)
assert "10." in msgs[0]
run.main()
assert len(msgs) == 2, "doppio invio!"
# giorno senza sessioni, gara tra 5 giorni -> riepilogo giornaliero
run.NOW = dt.datetime(2026, 5, 12, 9, 0, tzinfo=dt.timezone.utc)
db.write("quali", q); n0 = len(msgs); run.main()
assert len(msgs) == n0 + 1 and "tra 5 giorni" in msgs[-1]
run.main(); assert len(msgs) == n0 + 1
print("LOGIC TEST OK")
