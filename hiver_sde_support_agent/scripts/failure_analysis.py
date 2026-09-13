from pathlib import Path
import pandas as pd

p = Path("reports/predictions.csv")
if not p.exists():
    raise SystemExit("Run evaluate.py first.")
g = pd.read_csv("data/golden/golden_to_label.csv")
pred = pd.read_csv(p)
out = g.merge(pred, on="customer_tweet_id", suffixes=("_gold", "_pred"))
fail = out[out.gold_intent != out.pred_intent].sort_values("confidence")
fail.to_csv("reports/top_intent_failures.csv", index=False)
print(f"Intent failures: {len(fail)}. Wrote reports/top_intent_failures.csv")
print(fail[["customer_text", "gold_intent", "pred_intent", "confidence"]].head(10).to_string(index=False))
