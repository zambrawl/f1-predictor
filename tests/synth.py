"""Dati finti ma realistici (con nomi squadra che cambiano, tempi, pioggia) per i test."""
import datetime as dt
import numpy as np
import pandas as pd

TEAMS = {2023: ["red_bull", "ferrari", "mercedes", "mclaren", "aston_martin", "alpine", "williams",
                "toro_rosso", "sauber", "haas"],
         2024: ["red_bull", "ferrari", "mercedes", "mclaren", "aston_martin", "alpine", "williams",
                "alphatauri", "sauber", "haas"],
         2025: ["red_bull", "ferrari", "mercedes", "mclaren", "aston_martin", "alpine", "williams",
                "rb", "sauber", "haas"],
         2026: ["red_bull", "ferrari", "mercedes", "mclaren", "aston_martin", "alpine", "williams",
                "rb", "sauber", "haas"]}


def tstr(sec):
    return f"{int(sec // 60)}:{sec % 60:06.3f}"


def make(seed=1, rounds=8, seasons=(2023, 2024, 2025, 2026), with_next=True):
    rng = np.random.default_rng(seed)
    res, q, sp, w, fp = [], [], [], [], []
    for s in seasons:
        teams = TEAMS[s]
        for r in range(1, rounds + 1):
            date = dt.date(s, 3, 1) + dt.timedelta(days=14 * r)
            rain = float(rng.random() < .2) * 3
            strength = {}
            for i in range(20):
                strength[f"d{i}"] = -(i // 2) * 1.2 + rng.normal(0, 1.5) - (0 if rain == 0 else rng.normal(0, 1))
            qorder = sorted(strength, key=lambda d: strength[d] + rng.normal(0, 1), reverse=True)
            order = sorted(strength, key=lambda d: strength[d] + rng.normal(0, 1), reverse=True)
            base = 80 + rng.random() * 8
            for i in range(20):
                d = f"d{i}"
                code, team = f"D{i:02d}", teams[i // 2]
                qp, fp_ = qorder.index(d) + 1, order.index(d) + 1
                meta = dict(season=s, round=r, race_name=f"GP{r}", date=date.isoformat(), time="13:00:00Z",
                            lat=0.0, lon=0.0, circuit_id=f"c{r % 6}")
                res.append(dict(**meta, driver_id=d, code=code, team_id=team, grid=qp, finish=fp_,
                                status="Finished" if rng.random() > .08 else "Engine", points=0.0,
                                race_time="1:30:00.000" if fp_ == 1 else f"+{fp_ * 1.7:.1f}s"))
                t1 = base + qp * 0.12 + rng.normal(0, .05)
                q.append(dict(season=s, round=r, driver_id=d, code=code, team_id=team, qpos=qp, q1=tstr(t1),
                              q2=tstr(t1 - 0.6) if qp <= 15 else None, q3=tstr(t1 - 1.1) if qp <= 10 else None))
                if s >= 2024 and r % 4 == 0:
                    sp.append(dict(season=s, round=r, driver_id=d, sprint_pos=qp))
                if s >= 2023:
                    fp.append(dict(season=s, round=r, session="FP2", code=code, best_s=base + (i // 2) * .15 + rng.normal(0, .1),
                                   long_s=base + 4 + (i // 2) * .2 + rng.normal(0, .1)))
            w.append(dict(season=s, round=r, rain_mm=rain, temp=20.0))
    res, q, sp, w, fp = map(pd.DataFrame, (res, q, sp, w, fp))
    return res, q, sp, w, fp
