import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config
from src.retrieval import TfidfSupportIndex

c = load_config()
df = pd.read_csv(c["paths"]["pairs_csv"]).dropna(subset=["customer_text", "support_reply"])
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.sort_values("created_at")
# IMPORTANT: retrieval index only contains pre-cutoff development data; golden data is never indexed.
df = df.iloc[: int(len(df) * 0.80)].copy()
df = df.sample(min(len(df), 30000), random_state=42)
idx = TfidfSupportIndex(df.customer_text, df.support_reply, df.customer_tweet_id)
idx.save(c["paths"]["retrieval_index"])
print(f"Indexed {len(df):,} pre-cutoff historical examples")
