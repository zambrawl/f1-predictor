"""Da lanciare UNA volta in locale: scarica lo storico, addestra i modelli, stampa la validazione."""
import datetime as dt
from . import db, ingest, model
from .config import FIRST_SEASON
from .features import make_features


def main():
    ingest.refresh(range(FIRST_SEASON, dt.date.today().year + 1))
    df = make_features(db.read("results"), db.read("quali"), db.read("sprint"), db.read("weather"))
    model.evaluate(df)
    model.train_all(df)
    print("Fatto. Ora fai commit di data/ e models/.")


if __name__ == "__main__":
    main()
