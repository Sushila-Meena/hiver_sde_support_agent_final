import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config
from src.data import load_pairs

p = argparse.ArgumentParser()
p.add_argument("--input")
p.add_argument("--brand", default=None)
a = p.parse_args()
c = load_config()
brand = a.brand or c["brand"]["company_author_id"]
pairs = load_pairs(a.input or c["paths"]["raw_csv"], brand)
out = Path(c["paths"]["pairs_csv"])
out.parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(out, index=False)
print(f"Built {len(pairs):,} customer→{brand} response pairs")
print(f"Date range: {pairs.created_at.min()} → {pairs.created_at.max()}")
