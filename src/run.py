"""Orchestratore: lanciato ogni ora da GitHub Actions (ven-lun). Decide da solo cosa fare."""
import datetime as dt
import numpy as np
import pandas as pd
from . import db, ingest, jolpica, model, notify, practice, weather
from .config import ROOT
from .features import make_features

TODAY = dt.date.today()


# ---------- stato (per non inviare due volte la stessa notifica)
def sent(key):
    s = db.read("sent")
    return len(s) > 0 and key in set(s["key"])


def mark(key):
    db.upsert("sent", pd.DataFrame({"key": [key]}), ["key"])


# ---------- utilita'
def load_odds(s, r):
    """odds.csv: colonne season,round,code,odds (quote decimali). Si compila a mano prima della qualifica."""
    p = ROOT / "odds.csv"
    if not p.exists():
        return None
    o = pd.read_csv(p)
    o = o[(o["season"] == s) & (o["round"] == r)]
    return o if len(o) else None


def top_lines(cur, n=3):
    top = cur.sort_values("prob", ascending=False).head(n)
    return "\n".join(f"{i + 1}. {x.code}  {x.prob * 100:.1f}%" for i, x in enumerate(top.itertuples()))


def value_lines(cur, odds):
    o = odds.set_index("code")["odds"]
    imp = 1 / o
    if imp.sum() >= 0.9:                       # tolgo il margine del bookmaker
        imp = imp / imp.sum()
    x = cur.set_index("code")[["prob"]].join(imp.rename("imp"), how="inner")
    x["edge"] = x["prob"] - x["imp"]
    x = x[(x["edge"] > 0) & (x["prob"] >= 0.05)].sort_values("edge", ascending=False).head(3)
    if x.empty:
        return "Nessun pilota con valore positivo rispetto alle quote."
    return "\n".join(f"{i + 1}. {c}: modello {r.prob * 100:.0f}% vs quota {o[c]:.2f} "
                     f"(implicita {r.imp * 100:.0f}%, vantaggio +{r.edge * 100:.0f} pt)"
                     for i, (c, r) in enumerate(x.iterrows()))


def save_preds(cur, phase):
    p = cur[["season", "round", "driver_id", "code", "prob"]].copy()
    p["phase"] = phase
    db.upsert("predictions", p, ["season", "round", "phase", "driver_id"])


def predict_phase(nxt, phase):
    s, r = nxt["season"], nxt["round"]
    res, quali = db.read("results"), db.read("quali")
    sprint, wea = db.read("sprint"), db.read("weather")
    qn = quali[(quali["season"] == s) & (quali["round"] == r)]
    if phase == "post":     # griglia = ordine delle qualifiche (le penalita' non sono incluse)
        rows = qn.merge(res.drop_duplicates(["driver_id"], keep="last")[["driver_id", "code"]],
                        on="driver_id", how="left")
        team = res.sort_values(["season", "round"]).drop_duplicates("driver_id", keep="last")[["driver_id", "team_id"]]
        rows = rows.merge(team, on="driver_id", how="left").rename(columns={"qpos": "grid"})
        rows["code"] = rows["code"].fillna(rows["driver_id"].str[:3].str.upper())
        rows = rows.dropna(subset=["team_id"])
    else:                   # prima delle qualifiche: schieramento dell'ultima gara
        last = res[res["season"] == res["season"].max()]
        last = last[last["round"] == last["round"].max()]
        rows = last[["driver_id", "code", "team_id"]].copy()
        rows["grid"] = np.nan
    rows = rows[["driver_id", "code", "team_id", "grid"]].copy()
    for k in ("season", "round", "race_name", "date", "time", "lat", "lon", "circuit_id"):
        rows[k] = nxt[k]
    rows["finish"], rows["status"], rows["points"] = np.nan, None, np.nan
    rain, temp = weather.forecast(nxt)
    w = pd.concat([wea[~((wea["season"] == s) & (wea["round"] == r))],
                   pd.DataFrame([{"season": s, "round": r, "rain_mm": rain, "temp": temp}])])
    feat = make_features(pd.concat([res, rows], ignore_index=True), quali, sprint, w)
    cur = feat[(feat["season"] == s) & (feat["round"] == r)].copy()
    cur["prob"] = model.predict(cur, phase)
    return cur, rain


def rain_txt(rain):
    return "n.d." if pd.isna(rain) else f"{rain:.1f} mm nelle 3 ore di gara"


# ---------- flusso principale
def main():
    nxt = jolpica.next_race()
    ingest.refresh(sorted({TODAY.year, nxt["season"]} if nxt else {TODAY.year}))
    res, quali = db.read("results"), db.read("quali")
    retrain = False

    # A) gare finite: confronto previsione/risultato e riaddestramento
    preds = db.read("predictions")
    if len(preds):
        for (s, r), g in preds[preds["phase"] == "post"].groupby(["season", "round"]):
            if sent(f"result-{s}-{r}"):
                continue
            rr = res[(res["season"] == s) & (res["round"] == r)]
            if rr.empty:
                continue
            win = rr[rr["finish"] == 1].iloc[0]
            g = g.sort_values("prob", ascending=False).reset_index(drop=True)
            rank = g.index[g["code"] == win["code"]]
            rank_txt = f"{rank[0] + 1}o nella mia previsione ({g.loc[rank[0], 'prob'] * 100:.0f}%)" if len(rank) else "non previsto"
            notify.send(f"🏆 {win['race_name']}: ha vinto {win['code']} ({rank_txt}).\n"
                        f"La mia top 3 era:\n{top_lines(g, 3)}\n\n🔁 Modello riaddestrato con questa gara.")
            mark(f"result-{s}-{r}")
            retrain = True
    if retrain:
        df = make_features(res, quali, db.read("sprint"), db.read("weather"))
        model.train_all(df)

    # B) weekend in corso (venerdi' = 2 giorni prima, domenica = 0)
    if not nxt:
        return
    s, r = nxt["season"], nxt["round"]
    days = (dt.date.fromisoformat(nxt["date"]) - TODAY).days
    if not 0 <= days <= 2:
        return
    qn = quali[(quali["season"] == s) & (quali["round"] == r)]

    if len(qn):   # qualifiche ufficiali disponibili (cioe' sessione davvero conclusa, bandiere rosse incluse)
        if sent(f"post-{s}-{r}"):
            return
        cur, rain = predict_phase(nxt, "post")
        save_preds(cur, "post")
        msg = (f"🏁 {nxt['race_name']} – dopo le qualifiche\n"
               f"Meteo gara: {rain_txt(rain)}\n\n🥇 Top 3 per probabilita' di vittoria:\n{top_lines(cur)}")
        odds = load_odds(s, r)
        if odds is not None:
            msg += f"\n\n💰 Top 3 di valore (modello vs quote):\n{value_lines(cur, odds)}"
        else:
            msg += "\n\n(Nessuna quota in odds.csv: ti mostro solo le probabilita'.)"
        msg += "\n\nStrumento di analisi, non una garanzia. Gioca solo cio' che puoi permetterti di perdere."
        notify.send(msg)
        mark(f"post-{s}-{r}")

    elif not sent(f"pre-{s}-{r}"):   # stima preliminare dopo le libere
        pace = practice.long_run_pace(s, r)
        if pace is None and days == 2:
            return                   # libere non ancora disponibili: riprovo alla prossima ora
        cur, rain = predict_phase(nxt, "pre")
        save_preds(cur, "pre")
        msg = (f"🔧 {nxt['race_name']} – stima preliminare (prima delle qualifiche)\n"
               f"Meteo gara: {rain_txt(rain)}\n\n🥇 Top 3 provvisorio:\n{top_lines(cur)}")
        if pace is not None:
            name, p = pace
            lines = "\n".join(f"{i + 1}. {c}  +{v - p.iloc[0]:.2f}s" for i, (c, v) in enumerate(p.head(6).items()))
            msg += f"\n\n⏱ Ritmo long run {name} (giri puliti; carburante e gomme ignoti):\n{lines}"
        notify.send(msg)
        mark(f"pre-{s}-{r}")


if __name__ == "__main__":
    main()
