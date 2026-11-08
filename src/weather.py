"""Meteo con Open-Meteo (gratis, senza chiave): pioggia e temperatura nelle 3 ore della gara."""
import math
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
NAN = (math.nan, math.nan)


def _call(url, lat, lon, date, time_str):
    p = {"latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
         "hourly": "precipitation,temperature_2m", "timezone": "UTC"}
    h = requests.get(url, params=p, timeout=30).json()["hourly"]
    hr = int(str(time_str)[:2])
    idx = range(hr, min(hr + 3, 24))
    pr = [h["precipitation"][i] for i in idx]
    te = [h["temperature_2m"][i] for i in idx]
    if any(v is None for v in pr + te):
        return NAN
    return float(sum(pr)), float(sum(te) / len(te))


def at(lat, lon, date, time_str):
    """Meteo di una gara passata: prima l'archivio, se manca (gare recentissime) il forecast."""
    for url in (ARCHIVE, FORECAST):
        try:
            r = _call(url, lat, lon, date, time_str)
            if not math.isnan(r[0]):
                return r
        except Exception:
            pass
    return NAN


def forecast(nxt):
    try:
        return _call(FORECAST, nxt["lat"], nxt["lon"], nxt["date"], nxt["time"])
    except Exception:
        return NAN
