"""Da lanciare UNA volta: scarica lo storico dal 2014, valuta il modello, lo addestra.
Usa i dati delle prove libere (data/practice) se presenti."""
import datetime as dt
from . import db, ingest, model, practice
from .config import FIRST_SEASON
from .features import make_features


def main():
    ingest.refresh(range(FIRST_SEASON, dt.date.today().year + 1))
    df = make_features(db.read("results"), db.read("quali"), db.read("sprint"), db.read("weather"), practice.table())
    model.evaluate(df)
    model.train_all(df)
    model.mark_trained(df)
    print("Fatto. Ora fai commit di data/ e models/.")


if __name__ == "__main__":
    main()
