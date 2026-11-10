from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
DB = DATA / "f1.db"

FIRST_SEASON = 2014      # primo anno dello storico (era ibrida)
HALF_LIFE_RACES = 60     # le gare recenti pesano di piu (dimezza ogni 60 gare)
