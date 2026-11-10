"""Calcoli: feature, assenza di 'dati dal futuro', valutazione, addestramento, previsione."""
import numpy as np
from tests.synth import make
from src import model, practice, report
from src.features import make_features, PRE, POST
from src.config import DB

if DB.exists():
    DB.unlink()
res, q, sp, w, fp = make()
fpt = practice.aggregate(fp)
df = make_features(res, q, sp, w, fpt)

# 1) nessuna feature vuota dove non deve esserlo
for c in ["drv_pos5", "team_pos5", "drv_circ_pos", "drv_wet_gain", "exp_races", "drv_ytd_pos"]:
    assert not df[c].isna().any(), c
# 2) squadre rinominate = stessa macchina (toro_rosso -> alphatauri -> rb)
assert "toro_rosso" not in set(df["team_id"]) and "alphatauri" not in set(df["team_id"])
# 3) esperienza cresce, pilota nuovo parte dal valore della macchina
d0 = df[df["driver_id"] == "d0"].sort_values(["season", "round"])
assert d0["exp_races"].is_monotonic_increasing and d0["exp_races"].iloc[0] == 0
assert abs(d0["drv_pos5"].iloc[0] - d0["team_pos5"].iloc[0]) < 1e-9
# 4) NESSUN dato dal futuro: cambiare il risultato dell'ultima gara non deve cambiare le sue feature
last = (res["season"] == res["season"].max()) & (res["round"] == res["round"].max())
res2 = res.copy()
res2.loc[last, "finish"] = res2.loc[last, "finish"].values[::-1]
res2.loc[last, "status"] = "Engine"
df2 = make_features(res2, q, sp, w, fpt)
a = df[(df.season == 2026) & (df["round"] == 8)].set_index("driver_id").sort_index()
b = df2[(df2.season == 2026) & (df2["round"] == 8)].set_index("driver_id").sort_index()
for c in POST:
    assert np.allclose(a[c].astype(float).fillna(-999), b[c].astype(float).fillna(-999)), f"leak in {c}"
print("nessun dato dal futuro: OK")
# 5) una macchina forte ha un punteggio macchina piu' alto di una debole
cur = df[(df.season == 2026) & (df["round"] == 8)].copy()
sc = report.factor_scores(cur)
top_team = cur.loc[cur.driver_id == "d0"].index[0]; low_team = cur.loc[cur.driver_id == "d19"].index[0]
assert sc.loc[top_team, "Macchina"] > sc.loc[low_team, "Macchina"]
# 6) valutazione + addestramento + previsione normalizzata
model.evaluate(df, last_n=2)
model.train_all(df)
cur["prob"] = model.predict(cur, "post")
assert abs(cur["prob"].sum() - 1) < 1e-6
print(model.factor_weights("post"))
print("SMOKE TEST OK")
