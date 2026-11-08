import numpy as np
import pandas as pd
import xgboost as xgb
from .config import MODELS
from .features import PRE, POST

PHASES = {"pre": PRE, "post": POST}   # pre = dopo le libere (senza griglia), post = dopo le qualifiche


def _new():
    return xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, eval_metric="logloss")


def _norm(df, p):
    s = pd.Series(np.asarray(p), index=df.index)
    return s / s.groupby([df["season"], df["round"]]).transform("sum")   # somma 100% per gara


def train_all(df):
    d = df[df["finish"].notna()]
    for ph, feats in PHASES.items():
        m = _new()
        m.fit(d[feats], d["win"], sample_weight=d["w"])
        m.get_booster().save_model(str(MODELS / f"{ph}.json"))
    print(f"[model] riaddestrati {len(PHASES)} modelli su {d.groupby(['season', 'round']).ngroups} gare")


def predict(df, phase):
    b = xgb.Booster()
    b.load_model(str(MODELS / f"{phase}.json"))
    p = b.predict(xgb.DMatrix(df[PHASES[phase]]))
    return _norm(df, p)


def evaluate(df, last_n=3):
    """Validazione temporale: si addestra sulle stagioni precedenti e si testa su quella successiva."""
    seasons = sorted(df["season"].unique())[-last_n:]
    for ph, feats in PHASES.items():
        races = hit1 = hit3 = pole = 0
        ll = []
        for s in seasons:
            tr = df[(df["season"] < s) & df["finish"].notna()]
            te = df[(df["season"] == s) & df["finish"].notna()].copy()
            if tr.empty or te.empty:
                continue
            m = _new().fit(tr[feats], tr["win"], sample_weight=tr["w"])
            te["p"] = _norm(te, m.predict_proba(te[feats])[:, 1])
            for _, g in te.groupby("round"):
                if (g["finish"] == 1).sum() != 1:
                    continue
                races += 1
                top = g.sort_values("p", ascending=False)
                win = g[g["finish"] == 1].iloc[0]
                hit1 += int(top.iloc[0]["finish"] == 1)
                hit3 += int((top.head(3)["finish"] == 1).any())
                pole += int(win["grid"] == 1)
                ll.append(-np.log(max(win["p"], 1e-6)))
        if races:
            print(f"[valutazione fase '{ph}'] gare testate: {races} | "
                  f"vincitore al 1o posto: {hit1 / races:.0%} | nei primi 3: {hit3 / races:.0%} | "
                  f"(baseline 'vince il poleman': {pole / races:.0%}) | "
                  f"log-loss: {np.mean(ll):.2f} (casuale = {np.log(20):.2f}, piu' basso e' meglio)")
