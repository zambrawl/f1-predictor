"""Scarica lo storico delle prove libere (FastF1, dal 2018). Lungo: si puo' interrompere e rilanciare."""
import argparse
from . import practice

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, default=300)
    ap.add_argument("--since", type=int, default=2018)
    a = ap.parse_args()
    practice.build_history(a.max_minutes, a.since)
