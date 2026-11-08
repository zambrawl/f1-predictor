"""Test con dati finti: verifica che feature, addestramento, valutazione e previsione girino."""
import numpy as np, pandas as pd
from src import db, model
from src.features import make_features
from src.config import DB

if DB.exists():
    DB.unlink()
rng = np.random.default_rng(0)
teams = [f"t{i}" for i in range(10)]
drivers = [(f"d{i}", f"D{i:02d}", teams[i // 2]) for i in range(20)]
res, q, sp, w = [], [], [], []
for s in range(2018, 2024):
    for r in range(1, 21):
        strength = {d: -(int(d[1:]) // 2) * 1.2 + rng.normal(0, 1.5) for d, _, _ in drivers}
        order = sorted(strength, key=strength.get, reverse=True)
        grid = {d: i + 1 for i, d in enumerate(sorted(strength, key=lambda d: strength[d] + rng.normal(0, 1)))}
        for d, c, t in drivers:
            res.append(dict(season=s, round=r, race_name=f"GP{r}", date="x", time="13:00:00Z", lat=0, lon=0,
                            circuit_id=f"c{r}", driver_id=d, code=c, team_id=t, grid=grid[d],
                            finish=order.index(d) + 1, status="Finished" if rng.random() > .08 else "Engine", points=0.0))
            q.append(dict(season=s, round=r, driver_id=d, qpos=grid[d]))
            if s >= 2021 and r % 5 == 0:
                sp.append(dict(season=s, round=r, driver_id=d, sprint_pos=grid[d]))
        w.append(dict(season=s, round=r, rain_mm=float(rng.random() < .2) * 3, temp=20.0))
res, q, sp, w = map(pd.DataFrame, (res, q, sp, w))
df = make_features(res, q, sp, w)
assert not df[model.PHASES["pre"]].isna().any().any()
model.evaluate(df, last_n=2)
model.train_all(df)
cur = df[(df.season == 2023) & (df["round"] == 20)].copy()
cur["prob"] = model.predict(cur, "post")
print(cur.sort_values("prob", ascending=False)[["code", "grid", "prob"]].head(3))
assert abs(cur["prob"].sum() - 1) < 1e-6
print("SMOKE TEST OK")
