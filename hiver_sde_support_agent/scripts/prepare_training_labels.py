import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500)
a = p.parse_args()
c = load_config()
df = pd.read_csv(c["paths"]["pairs_csv"]).dropna(subset=["customer_text"]).drop_duplicates("customer_text")
# Time-aware split: final 20% is reserved for the untouched golden set.
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.sort_values("created_at").reset_index(drop=True)
cut = int(len(df) * 0.80)
train_pool = df.iloc[:cut].copy()
# Sample across time to avoid only labeling one period.
if len(train_pool) > a.n:
    train_pool = train_pool.sample(a.n, random_state=2026)
out = train_pool[["customer_tweet_id", "customer_text", "support_reply", "created_at"]].copy()
out["intent"] = ""
out["label_notes"] = ""
Path(c["paths"]["train_labels"]).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(c["paths"]["train_labels"], index=False)
print(f"Created {len(out)} training-label candidates: {c['paths']['train_labels']}")
print("Label these manually using configs/intent_taxonomy.yaml after reviewing intent_clusters.csv.")
