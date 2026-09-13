import argparse, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import load_config
from src.intent import load, predict
from src.retrieval import TfidfSupportIndex
from src.agent import draft_reply
from src.judge import judge
from src.evaluation import pearson_or_none

p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=30); a = p.parse_args()
c = load_config(); g = pd.read_csv(c["paths"]["golden"]).sample(min(a.n, len(pd.read_csv(c["paths"]["golden"]))), random_state=7)
m = load(c["paths"]["model"]); idx = TfidfSupportIndex.load(c["paths"]["retrieval_index"])
rows = []
for _, r in g.iterrows():
    it, _, _ = predict(m, r.customer_text); ev = idx.search(r.customer_text, 5); d = draft_reply(r.customer_text, it, ev); j = judge(r.customer_text, d["reply"], ev)
    rows.append({"customer_tweet_id": r.customer_tweet_id, "customer_text": r.customer_text, "draft_reply": d["reply"], "human_reply_quality": "", "human_notes": "", "judge_overall": j["overall"], "judge_reason": j["reason"]})
out = pd.DataFrame(rows); out.to_csv("data/golden/judge_calibration.csv", index=False)
print("Created data/golden/judge_calibration.csv. Human-label human_reply_quality (1-5) before reporting judge agreement.")
