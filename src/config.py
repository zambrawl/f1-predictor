from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
DB = DATA / "f1.db"

FIRST_SEASON = 2018      # primo anno dello storico
HALF_LIFE_RACES = 40     # le gare piu' recenti pesano di piu' (dimezza ogni 40 gare)
