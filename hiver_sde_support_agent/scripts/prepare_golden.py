import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=200)
a = p.parse_args()
c = load_config()
df = pd.read_csv(c["paths"]["pairs_csv"]).dropna(subset=["customer_text"]).drop_duplicates("customer_text")
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.sort_values("created_at").reset_index(drop=True)
cut = int(len(df) * 0.80)
golden_pool = df.iloc[cut:].copy()
# Fixed random sample from future/held-out data. This file must remain untouched during training.
out = golden_pool.sample(min(a.n, len(golden_pool)), random_state=2026)[
    ["customer_tweet_id", "customer_text", "support_reply", "created_at"]
].copy()
out["gold_intent"] = ""
out["gold_auto_handle"] = ""
out["gold_reply_quality"] = ""
out["gold_notes"] = ""
Path(c["paths"]["golden"]).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(c["paths"]["golden"], index=False)
print(f"Created {len(out)} held-out golden examples: {c['paths']['golden']}")
print("Human-label this file. Do not train on it or include it in retrieval index.")
