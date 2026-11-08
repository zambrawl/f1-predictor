"""Orchestratore: gira ogni ora (ogni 15 min nel weekend) e decide da solo cosa fare."""
import datetime as dt
import os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from . import db, grid, ingest, jolpica, model, notify, weather
from .config import ROOT
from .features import make_features

ROME = ZoneInfo("Europe/Rome")
NOW = dt.datetime.now(dt.timezone.utc)
TOP_N = 10
DAILY_HOUR = 8            # ora italiana del riepilogo giornaliero
REMINDER_MIN = 30         # promemoria a 30 minuti dalla sessione


# ---------- stato (per non inviare due volte la stessa notifica)
def sent(key):
    s = db.read("sent")
    return len(s) > 0 and key in set(s["key"])


def mark(key):
    db.upsert("sent", pd.DataFrame({"key": [key]}), ["key"])


# ---------- utilita'
def load_odds(s, r):
    """odds.csv: colonne season,round,code,odds (quote decimali)."""
    p = ROOT / "odds.csv"
    if not p.exists():
        return None
    o = pd.read_csv(p)
    o = o[(o["season"] == s) & (o["round"] == r)]
    return o if len(o) else None


def top_lines(cur, n=TOP_N):
    top = cur.sort_values("prob", ascending=False).head(n)
    return "\n".join(f"{i + 1}. {x.code}  {x.prob * 100:.1f}%" for i, x in enumerate(top.itertuples()))


def value_lines(cur, odds):
    o = odds.set_index("code")["odds"]
    imp = 1 / o
    if imp.sum() >= 0.9:                       # tolgo il margine del bookmaker
        imp = imp / imp.sum()
    x = cur.set_index("code")[["prob"]].join(imp.rename("imp"), how="inner")
    x["edge"] = x["prob"] - x["imp"]
    x = x[(x["edge"] > 0) & (x["prob"] >= 0.02)].sort_values("edge", ascending=False).head(TOP_N)
    if x.empty:
        return "Nessun pilota con valore positivo rispetto alle quote."
    return "\n".join(f"{i + 1}. {c}: modello {r.prob * 100:.0f}% vs quota {o[c]:.2f} "
                     f"(implicita {r.imp * 100:.0f}%, +{r.edge * 100:.0f} pt)"
                     for i, (c, r) in enumerate(x.iterrows()))


def save_preds(cur, phase):
    p = cur[["season", "round", "driver_id", "code", "prob"]].copy()
    p["phase"] = phase
    db.upsert("predictions", p, ["season", "round", "phase", "driver_id"])


def predict_phase(nxt, phase):
    """Ritorna (previsioni, pioggia prevista, fonte della griglia)."""
    s, r = nxt["season"], nxt["round"]
    res, quali = db.read("results"), db.read("quali")
    sprint, wea = db.read("sprint"), db.read("weather")
    src = "n.d. (prima delle qualifiche)"
    if phase == "post":
        qn = quali[(quali["season"] == s) & (quali["round"] == r)]
        info = (res.sort_values(["season", "round"]).drop_duplicates("driver_id", keep="last")
                [["driver_id", "code", "team_id"]])
        rows = qn.merge(info, on="driver_id", how="left").rename(columns={"qpos": "grid"})
        rows["code"] = rows["code"].fillna(rows["driver_id"].str[:3].str.upper())
        rows = rows.dropna(subset=["team_id"])
        new_grid, src = grid.resolve(s, r, rows)
        rows["grid"] = new_grid.fillna(rows["grid"])
    else:
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
    return cur, rain, src


def rain_txt(rain):
    return "n.d." if pd.isna(rain) else f"{rain:.1f} mm nelle 3 ore di gara"


def body(cur, rain, src, nxt):
    msg = f"Meteo gara: {rain_txt(rain)}\nGriglia: {src}\n\n🥇 Top {TOP_N} per probabilita' di vittoria:\n{top_lines(cur)}"
    odds = load_odds(nxt["season"], nxt["round"])
    if odds is not None:
        msg += f"\n\n💰 Migliori {TOP_N} di valore (modello vs quote):\n{value_lines(cur, odds)}"
    return msg


# ---------- comandi Telegram (/classifica) e invio classifica a richiesta
COMMANDS = {"/classifica", "/top", "/previsione"}


def send_ranking(nxt):
    if not nxt:
        notify.send("Nessuna gara in programma al momento.")
        return
    s, r = nxt["season"], nxt["round"]
    quali = db.read("quali")
    has_quali = len(quali[(quali["season"] == s) & (quali["round"] == r)]) > 0
    cur, rain, src = predict_phase(nxt, "post" if has_quali else "pre")
    stato = "dopo le qualifiche" if has_quali else "prima delle qualifiche"
    notify.send(f"📊 Classifica attuale – {nxt['race_name']} ({stato})\n{body(cur, rain, src, nxt)}")


def handle_commands(nxt):
    """Legge i messaggi arrivati al bot: se scrivi /classifica (dalla tua chat), rispondo."""
    chat = os.getenv("TELEGRAM_CHAT_ID")
    old = db.read("tg")
    offset = int(old["offset"].iloc[0]) + 1 if len(old) else None
    ups = notify.get_updates(offset)
    if not ups:
        return False
    wanted = False
    for u in ups:
        m = u.get("message") or {}
        text = (m.get("text") or "").strip()
        if str(m.get("chat", {}).get("id")) == str(chat) and text:
            if text.split()[0].split("@")[0].lower() in COMMANDS:
                wanted = True
    db.write("tg", pd.DataFrame({"offset": [max(u["update_id"] for u in ups)]}))
    if wanted:
        send_ranking(nxt)
    return wanted


# ---------- flusso principale
def main():
    nxt = jolpica.next_race()
    ingest.refresh(sorted({NOW.year, nxt["season"]} if nxt else {NOW.year}))
    res, quali = db.read("results"), db.read("quali")

    # 0) comandi Telegram o richiesta manuale dal pulsante "Run workflow"
    try:
        asked = handle_commands(nxt)
    except Exception as e:
        print("[comandi] errore:", e)
        asked = False
    if os.getenv("FORCE_RANKING") == "true" and not asked:
        send_ranking(nxt)

    # A) gare finite: confronto previsione/risultato e riaddestramento
    retrain = False
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
            rank_txt = (f"{rank[0] + 1}o nella mia previsione ({g.loc[rank[0], 'prob'] * 100:.0f}%)"
                        if len(rank) else "non previsto")
            notify.send(f"🏆 {win['race_name']}: ha vinto {win['code']} ({rank_txt}).\n"
                        f"La mia top {TOP_N} era:\n{top_lines(g)}\n\n🔁 Modello riaddestrato con questa gara.")
            mark(f"result-{s}-{r}")
            retrain = True
    if retrain:
        model.train_all(make_features(res, quali, db.read("sprint"), db.read("weather")))

    if not nxt:
        return
    s, r = nxt["season"], nxt["round"]
    has_quali = len(quali[(quali["season"] == s) & (quali["round"] == r)]) > 0

    # B) qualifiche ufficiali appena disponibili (sessione davvero conclusa, bandiere rosse incluse)
    if has_quali and not sent(f"post-{s}-{r}"):
        cur, rain, src = predict_phase(nxt, "post")
        save_preds(cur, "post")
        notify.send(f"🏁 {nxt['race_name']} – dopo le qualifiche\n{body(cur, rain, src, nxt)}")
        mark(f"post-{s}-{r}")

    # C) promemoria 30 minuti prima di ogni sessione (prove libere, sprint, qualifiche, gara)
    for label, start in nxt["sessions"]:
        mins = (start - NOW).total_seconds() / 60
        key = f"sess-{s}-{r}-{label}"
        if 0 < mins <= REMINDER_MIN and not sent(key):
            cur, rain, src = predict_phase(nxt, "post" if has_quali else "pre")
            if has_quali:
                save_preds(cur, "post")
            when = start.astimezone(ROME).strftime("%H:%M")
            notify.send(f"⏱ Tra circa {round(mins)} minuti: {label} – {nxt['race_name']} (ore {when})\n"
                        f"{body(cur, rain, src, nxt)}")
            mark(key)

    # D) riepilogo giornaliero (solo nei giorni senza sessioni, entro 14 giorni dalla gara)
    local = NOW.astimezone(ROME)
    session_today = any(st.astimezone(ROME).date() == local.date() for _, st in nxt["sessions"])
    days_left = (dt.date.fromisoformat(nxt["date"]) - local.date()).days
    key = f"daily-{local.date()}"
    if not session_today and 0 < days_left <= 14 and local.hour >= DAILY_HOUR and not sent(key):
        cur, rain, src = predict_phase(nxt, "pre")
        notify.send(f"📅 {nxt['race_name']} tra {days_left} giorni – stima di oggi\n{body(cur, rain, src, nxt)}")
        mark(key)


if __name__ == "__main__":
    main()
