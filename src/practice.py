"""Ritmo sui long run delle libere con FastF1 (informativo, non entra nel modello).
Esclude giri ai box e giri sotto safety car / VSC / bandiera rossa (TrackStatus != '1')."""
import tempfile


def long_run_pace(season, rnd):
    try:
        import fastf1
        fastf1.Cache.enable_cache(tempfile.mkdtemp())
    except Exception:
        return None
    for name in ("FP2", "FP1"):
        try:
            s = fastf1.get_session(season, rnd, name)
            s.load(telemetry=False, weather=False, messages=False)
            laps = s.laps
            laps = laps[laps["LapTime"].notna() & laps["PitInTime"].isna()
                        & laps["PitOutTime"].isna() & (laps["TrackStatus"].astype(str) == "1")].copy()
            if laps.empty:
                continue
            laps = laps[laps["LapTime"] < laps["LapTime"].min() * 1.07]
            laps["sec"] = laps["LapTime"].dt.total_seconds()
            st = laps.groupby(["Driver", "Stint"])["sec"].agg(["count", "median"]).reset_index()
            st = st[st["count"] >= 5]                       # solo stint lunghi (>=5 giri puliti)
            if st.empty:
                continue
            return name, st.groupby("Driver")["median"].min().sort_values()
        except Exception:
            continue
    return None
