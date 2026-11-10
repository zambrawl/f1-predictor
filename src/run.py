"""Orchestratore. Gira ogni 5 minuti (ven-dom) / 15 minuti (lun-gio) su GitHub Actions.
- risponde ai comandi Telegram (/classificagara, /classificaqualifiche, /penalita, /aiuto)
- notifiche automatiche SOLO venerdi', sabato e domenica: promemoria 30 min prima di ogni sessione,
  report dopo le qualifiche, esito della gara
- riaddestra il modello dopo ogni gara"""
import datetime as dt
import os
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from . import bot, db, grid, ingest, jolpica, model, notify, practice, report, weather
from .config import ROOT
from .features import make_features

ROME = ZoneInfo("Europe/Rome")
NOW = dt.datetime.now(dt.timezone.utc)
REMINDER_MIN = 30
NOTIFY_DAYS = {4, 5, 6}                     # venerdi' sabato domenica (lunedi' = 0)
HEART = ROOT / "data" / "heartbeat.txt"


# ---------- stato
def sent(key):
    s = db.read("sent")
    return len(s) > 0 and key in set(s["key"])


def mark(key):
    db.upsert("sent", pd.DataFrame({"key": [key]}), ["key"])


def notify_day():
    return NOW.astimezone(ROME).weekday() in NOTIFY_DAYS


# ---------- quote (facoltative)
def load_odds(s, r):
    p = ROOT / "odds.csv"
    if not p.exists():
        return None
    o = pd.read_csv(p)
    o = o[(o["season"] == s) & (o["round"] == r)]
    return o if len(o) else None


def value_lines(cur, odds):
    o = odds.set_index("code")["odds"]
    imp = 1 / o
    if imp.sum() >= 0.9:
        imp = imp / imp.sum()
    x = cur.set_index("code")[["prob"]].join(imp.rename("imp"), how="inner")
    x["edge"] = x["prob"] - x["imp"]
    x = x[(x["edge"] > 0) & (x["prob"] >= 0.02)].sort_values("edge", ascending=False).head(10)
    if x.empty:
        return "Nessun pilota con valore positivo rispetto alle quote."
    return "\n".join(f"{i + 1}. {c}: modello {r.prob * 100:.0f}% vs quota {o[c]:.2f} (+{r.edge * 100:.0f} pt)"
                     for i, (c, r) in enumerate(x.iterrows()))


# ---------- previsioni
def fill_info(q, res):
    """Completa codice e squadra dei piloti (utile per righe vecchie o piloti nuovi)."""
    info = (res.sort_values(["season", "round"]).drop_duplicates("driver_id", keep="last")
            [["driver_id", "code", "team_id"]].rename(columns={"code": "code_h", "team_id": "team_h"}))
    q = q.merge(info, on="driver_id", how="left")
    for c, h in (("code", "code_h"), ("team_id", "team_h")):
        if c not in q:
            q[c] = np.nan
        q[c] = q[c].fillna(q[h])
    q["code"] = q["code"].fillna(q["driver_id"].str[:3].str.upper())
    q["team_id"] = q["team_id"].fillna("unknown")
    return q.drop(columns=["code_h", "team_h"])


def has_quali(nxt, quali):
    return len(quali[(quali["season"] == nxt["season"]) & (quali["round"] == nxt["round"])]) > 0


def predict_phase(nxt, phase):
    """Ritorna (previsioni, pioggia prevista, fonte griglia, penalita')."""
    s, r = nxt["season"], nxt["round"]
    res, quali = db.read("results"), db.read("quali")
    sprint, wea = db.read("sprint"), db.read("weather")
    pens = []
    src = "non ancora disponibile (prima delle qualifiche)"
    if phase == "post":
        qn = fill_info(quali[(quali["season"] == s) & (quali["round"] == r)], res)
        rows = qn[["driver_id", "code", "team_id", "qpos"]].rename(columns={"qpos": "grid"})
        new_grid, src, pens = grid.resolve(s, r, rows)
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
    feat = make_features(pd.concat([res, rows], ignore_index=True), quali, sprint, w, practice.table())
    cur = feat[(feat["season"] == s) & (feat["round"] == r)].copy()
    cur["prob"] = model.predict(cur, phase)
    return cur, rain, src, pens


def save_preds(cur, phase):
    p = cur[["season", "round", "driver_id", "code", "prob"]].copy()
    p["phase"] = phase
    db.upsert("predictions", p, ["season", "round", "phase", "driver_id"])


# ---------- risposte ai comandi
def last_race(res):
    last = res.sort_values(["season", "round"]).iloc[-1]
    days = (NOW.date() - pd.Timestamp(last["date"]).date()).days
    return last, days


def cmd_race(nxt):
    res, quali = db.read("results"), db.read("quali")
    if len(res):
        last, ds = last_race(res)
        if 0 <= ds <= 2:        # gara appena finita: mostro i tempi ufficiali
            rr = res[(res["season"] == last["season"]) & (res["round"] == last["round"])]
            return report.result_text(rr, last["race_name"])
    if not nxt:
        return "Nessuna gara in programma al momento."
    post = has_quali(nxt, quali)
    phase = "post" if post else "pre"
    cur, rain, src, pens = predict_phase(nxt, phase)
    odds = load_odds(nxt["season"], nxt["round"])
    return report.race_prediction_text(
        cur, nxt["race_name"], "dopo le qualifiche" if post else "prima delle qualifiche",
        rain, src, pens, model.factor_weights(phase), value_lines(cur, odds) if odds is not None else None)


def cmd_quali(nxt):
    res, quali = db.read("results"), db.read("quali")
    if nxt and has_quali(nxt, quali):
        q = fill_info(quali[(quali["season"] == nxt["season"]) & (quali["round"] == nxt["round"])], res)
        return report.quali_text(q, nxt["race_name"])
    if len(res):
        last, ds = last_race(res)
        if 0 <= ds <= 3:
            q = quali[(quali["season"] == last["season"]) & (quali["round"] == last["round"])]
            if len(q):
                return report.quali_text(fill_info(q, res), last["race_name"])
    if nxt:
        fp = practice.table()
        fp = fp[(fp["season"] == nxt["season"]) & (fp["round"] == nxt["round"])]
        if len(fp):
            return report.practice_text(fp, nxt["race_name"])
        return f"Qualifiche di {nxt['race_name']} non ancora disputate e nessun dato dalle libere."
    return "Nessuna qualifica disponibile."


def cmd_penalty(args, nxt):
    res, quali = db.read("results"), db.read("quali")
    last = res[res["season"] == res["season"].max()]
    lineup = last[last["round"] == last["round"].max()][["driver_id", "code"]]
    if nxt and has_quali(nxt, quali):
        q = fill_info(quali[(quali["season"] == nxt["season"]) & (quali["round"] == nxt["round"])], res)
        lineup = pd.concat([q[["driver_id", "code"]], lineup]).drop_duplicates("driver_id")
    return bot.penalty_command(args, nxt, lineup)


def reply_for(cmd, args, nxt):
    if cmd in ("/classificagara", "/gara"):
        return cmd_race(nxt)
    if cmd in ("/classificaqualifiche", "/qualifiche"):
        return cmd_quali(nxt)
    if cmd == "/penalita":
        return cmd_penalty(args, nxt)
    return bot.HELP


def handle_commands(nxt):
    chat = os.getenv("TELEGRAM_CHAT_ID")
    old = db.read("tg")
    offset = int(old["offset"].iloc[0]) + 1 if len(old) else None
    ups = notify.get_updates(offset)
    if not ups:
        return
    db.write("tg", pd.DataFrame({"offset": [max(u["update_id"] for u in ups)]}))
    seen = set()
    for u in ups:
        m = u.get("message") or {}
        text = (m.get("text") or "").strip()
        if str(m.get("chat", {}).get("id")) != str(chat) or not text.startswith("/"):
            continue
        cmd, args = bot.parse(text)
        if cmd != "/penalita":
            if (cmd, tuple(args)) in seen:
                continue
            seen.add((cmd, tuple(args)))
        try:
            notify.send(reply_for(cmd, args, nxt))
        except Exception as e:
            notify.send(f"⚠️ Non sono riuscito a rispondere a {cmd}: {e}")


# ---------- flusso principale
def _main():
    nxt = jolpica.next_race()
    full_ok = not sent(f"full-{NOW.date()}")
    if ingest.refresh_smart(nxt, NOW, full_ok) == "full":
        mark(f"full-{NOW.date()}")
    if nxt:
        practice.update_current(nxt, NOW, sent, mark)

    try:
        handle_commands(nxt)
    except Exception as e:
        print("[comandi] errore:", e)

    res, quali = db.read("results"), db.read("quali")
    if res.empty:
        return nxt

    # riaddestramento silenzioso dopo ogni gara nuova
    meta = db.read("meta")
    trained = int(meta["v"].iloc[0]) if len(meta) else 0
    if int((res["season"] * 100 + res["round"]).max()) > trained:
        df = make_features(res, quali, db.read("sprint"), db.read("weather"), practice.table())
        model.train_all(df)
        model.mark_trained(df)

    # --- da qui solo notifiche automatiche: SOLO venerdi', sabato, domenica
    if not notify_day():
        return nxt

    # esito delle gare previste
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
            txt = (f"{rank[0] + 1}o nella mia previsione ({g.loc[rank[0], 'prob'] * 100:.0f}%)"
                   if len(rank) else "non previsto")
            top = "\n".join(f"{i + 1}. {x.code}  {x.prob * 100:.1f}%" for i, x in enumerate(g.head(10).itertuples()))
            notify.send(f"🏆 {win['race_name']}: ha vinto {win['code']} ({txt}).\nLa mia top 10 era:\n{top}\n\n"
                        f"Scrivi /classificagara per i tempi ufficiali.")
            mark(f"result-{s}-{r}")

    if not nxt:
        return nxt
    s, r = nxt["season"], nxt["round"]
    post = has_quali(nxt, quali)

    # qualifiche ufficiali appena disponibili (sessione davvero conclusa, bandiere rosse incluse)
    if post and not sent(f"post-{s}-{r}"):
        cur, rain, src, pens = predict_phase(nxt, "post")
        save_preds(cur, "post")
        notify.send(f"🏁 {nxt['race_name']} – dopo le qualifiche\nMeteo gara: {report.rain_txt(rain)}\n"
                    f"Griglia: {src}\n\n🥇 Top 10 per probabilita' di vittoria:\n{report.compact_top(cur)}\n\n"
                    f"Dettaglio: /classificagara · penalita': /penalita VER 3")
        mark(f"post-{s}-{r}")

    # promemoria 30 minuti prima di ogni sessione
    for label, start, kind in nxt["sessions"]:
        mins = (start - NOW).total_seconds() / 60
        key = f"sess-{s}-{r}-{kind}"
        if 0 < mins <= REMINDER_MIN and not sent(key):
            cur, rain, src, pens = predict_phase(nxt, "post" if post else "pre")
            when = start.astimezone(ROME).strftime("%H:%M")
            notify.send(f"⏰ Tra circa {round(mins)} minuti: {label} – {nxt['race_name']} (ore {when})\n"
                        f"Meteo gara: {report.rain_txt(rain)}\n\n🥇 Top 10 vittoria:\n{report.compact_top(cur)}\n\n"
                        f"Dettaglio: /classificagara")
            mark(key)
    return nxt


def heartbeat():
    """Un piccolo commit al mese, cosi' GitHub non disattiva i workflow schedulati per inattivita'."""
    month = NOW.strftime("%Y-%m")
    if not HEART.exists() or HEART.read_text().strip() != month:
        HEART.write_text(month)


def main():
    _main()
    heartbeat()


if __name__ == "__main__":
    main()
