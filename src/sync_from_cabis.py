"""Copy the data snapshot from the CABIS repo into data/raw (one-way, read-only on the source).

Usage:
  python src/sync_from_cabis.py                 # default CABIS location (sibling folder on Desktop)
  python src/sync_from_cabis.py --cabis D:\\path  # custom location
Does NOT run any CABIS collector; it only copies files that already exist there.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FILES = {
    "market/clean_daily.csv": "clean_daily.csv",
    "app/data/garak_daily.csv": "garak_daily.csv",
    "app/data/weather_auto_daily.csv": "weather_auto_daily.csv",
    "app/data/stock.csv": "stock.csv",
    "app/data/history.csv": "history.csv",
    "app/data/at_gov_daily.csv": "at_gov_daily.csv",
    "app/data/produce.csv": "produce.csv",
    "app/data/guide.csv": "guide.csv",
    "market/outputs/krei_baechu_monthly_series_2022_2025.xlsx": "krei_baechu_monthly_series_2022_2025.xlsx",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cabis", default=str(ROOT.parent / "CABIS"), help="CABIS repo root")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src_root = Path(a.cabis)
    if not (src_root / "app").exists():
        raise SystemExit(f"CABIS root not found: {src_root}")
    RAW.mkdir(parents=True, exist_ok=True)
    for rel, name in FILES.items():
        s, d = src_root / rel, RAW / name
        if not s.exists():
            print(f"MISSING  {s}")
            continue
        same = d.exists() and d.stat().st_size == s.stat().st_size and int(d.stat().st_mtime) >= int(s.stat().st_mtime)
        tag = "same   " if same else ("would copy" if a.dry_run else "copied ")
        if not same and not a.dry_run:
            shutil.copy2(s, d)
        print(f"{tag}  {rel} -> data/raw/{name}  ({s.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
