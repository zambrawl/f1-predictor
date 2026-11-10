"""Testi dei messaggi Telegram."""
import numpy as np
import pandas as pd
from .features import team_name, _secs

ORDER = ["Macchina", "Pilota", "Esperienza", "Pioggia", "Pista", "Qualifica", "Libere", "Affidabilita"]


def _score(s, higher=True):
    n = s.notna().sum()
    if n < 2:
        return pd.Series(np.nan, index=s.index)
    r = s.rank(method="average")
    return (r - 1) / (n - 1) * 100 if higher else (n - r) / (n - 1) * 100


def _mean(*series):
    return pd.concat(series, axis=1).mean(axis=1)


def factor_scores(c):
    """Punteggi 0-100 per ogni fattore, rispetto agli altri piloti della gara (100 = il migliore)."""
    S = pd.DataFrame(index=c.index)
    S["Macchina"] = _mean(_score(c["team_pos5"], False), _score(c["team_ytd_pos"], False),
                          _score(c["team_qgap5"], False))
    S["Pilota"] = _mean(_score(c["drv_mate_delta5"], False), _score(c["drv_pos5"] - c["team_pos5"], False))
    S["Esperienza"] = (c["exp_races"] / 200 * 100).clip(upper=100)
    S["Pioggia"] = _score(c["drv_wet_gain"], True)
    S["Pista"] = _score(c["drv_circ_pos"], False)
    if c["grid"].notna().any():
        S["Qualifica"] = _mean(_score(c["grid"], False), _score(c["qgap"], False))
    else:
        S["Qualifica"] = _score(c["drv_qgap5"], False)
    S["Libere"] = _mean(_score(c["fp_best_gap"], False), _score(c["fp_long_gap"], False))
    S["Affidabilita"] = _score(c["drv_dnf10"], False)
    return S


def _f(v):
    return "n.d." if pd.isna(v) else f"{v:.0f}"


def compact_top(cur, n=10):
    top = cur.sort_values("prob", ascending=False).head(n)
    return "\n".join(f"{i}. {r['code']} ({team_name(r['team_id'])})  {r['prob'] * 100:.1f}%"
                     for i, (_, r) in enumerate(top.iterrows(), 1))


def rain_txt(rain):
    return "n.d." if pd.isna(rain) else f"{rain:.1f} mm nelle 3 ore di gara"


def pens_txt(pens):
    if not pens:
        return ""
    return "Penalita applicate: " + ", ".join(f"{c} " + ("dai box" if p >= 99 else f"-{int(p)}") for c, p in pens)


def race_prediction_text(cur, race_name, stato, rain, src, pens, weights, odds_txt=None, n=10):
    sc = factor_scores(cur)
    top = cur.sort_values("prob", ascending=False).head(n)
    lines = [f"📊 Pronostico gara – {race_name} ({stato})",
             f"Meteo gara: {rain_txt(rain)}", f"Griglia: {src}"]
    if pens:
        lines.append(pens_txt(pens))
    lines.append("Punteggi 0-100 rispetto agli altri piloti (100 = il migliore). "
                 "Le percentuali sommano 100% sull'intera griglia.\n")
    for i, (idx, r) in enumerate(top.iterrows(), 1):
        lines.append(f"{i}. {r['code']} ({team_name(r['team_id'])}) – {r['prob'] * 100:.1f}%")
        lines.append("   " + " · ".join(f"{k} {_f(sc.loc[idx, k])}" for k in ORDER))
    if weights:
        lines.append("\nPeso dei fattori nel modello: " + " · ".join(f"{k} {v:.0f}%" for k, v in weights.items()))
    if odds_txt:
        lines.append(f"\n💰 Migliori di valore (modello vs quote):\n{odds_txt}")
    return "\n".join(lines)


def _best(row):
    for tag in ("q3", "q2", "q1"):
        v = row.get(tag)
        if isinstance(v, str) and v:
            return v, tag.upper()
    return None, ""


def quali_text(q, title):
    q = q.sort_values("qpos")
    rows = [r for _, r in q.iterrows()]
    pole_t, _ = _best(rows[0]) if rows else (None, "")
    pole_s = _secs(pole_t) if pole_t else np.nan
    lines = [f"⏱ Qualifiche – {title} (tempi ufficiali)"]
    for r in rows:
        t, tag = _best(r)
        if t is None:
            txt = "n.d."
        elif r["qpos"] == rows[0]["qpos"]:
            txt = f"{t} (pole)"
        elif tag == "Q3" and not np.isnan(pole_s):
            txt = f"{t} (+{_secs(t) - pole_s:.3f})"
        else:
            txt = f"{t} ({tag})"
        lines.append(f"{int(r['qpos']):>2}. {r['code']} ({team_name(r['team_id'])})  {txt}")
    return "\n".join(lines)


def result_text(rr, title):
    rr = rr.sort_values("finish")
    if "race_time" not in rr:
        rr = rr.assign(race_time=None)
    lines = [f"🏁 Risultato gara – {title} (tempi ufficiali)"]
    for r in rr.itertuples():
        t = r.race_time if isinstance(r.race_time, str) and r.race_time else r.status
        g = f" · griglia {int(r.grid)}" if pd.notna(r.grid) and r.grid > 0 else " · dai box"
        lines.append(f"{int(r.finish):>2}. {r.code} ({team_name(r.team_id)})  {t}{g}")
    return "\n".join(lines)


def practice_text(fp, title):
    fp = fp.sort_values("fp_best_gap").head(10)
    lines = [f"Qualifiche di {title} non ancora disputate.", "⏱ Ritmo nelle prove libere (distacco % dal migliore):"]
    for i, r in enumerate(fp.itertuples(), 1):
        lp = "n.d." if pd.isna(r.fp_long_gap) else f"+{r.fp_long_gap:.2f}%"
        lines.append(f"{i}. {r.code}  giro secco +{r.fp_best_gap:.2f}% · passo gara {lp}")
    return "\n".join(lines)
