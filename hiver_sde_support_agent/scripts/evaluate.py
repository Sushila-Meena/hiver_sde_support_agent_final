import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.config import load_config
from src.intent import load, predict
from src.retrieval import TfidfSupportIndex
from src.routing import should_escalate

c = load_config()
g = pd.read_csv(c["paths"]["golden"])
if g.gold_intent.fillna("").str.strip().eq("").any():
    raise ValueError("Human-label gold_intent first.")
if g.gold_auto_handle.fillna("").str.strip().eq("").any():
    raise ValueError("Human-label gold_auto_handle first.")

m = load(c["paths"]["model"])
idx = TfidfSupportIndex.load(c["paths"]["retrieval_index"])
rows = []
for _, r in g.iterrows():
    intent, conf, _ = predict(m, r.customer_text)
    ev = idx.search(r.customer_text, c["model"]["retrieval_top_k"])
    esc, reason = should_escalate(r.customer_text, conf, ev, c["model"]["confidence_threshold"], c["routing"]["high_risk_patterns"])
    rows.append({
        "customer_tweet_id": r.customer_tweet_id,
        "gold_intent": r.gold_intent,
        "pred_intent": intent,
        "confidence": conf,
        "gold_auto_handle": r.gold_auto_handle,
        "pred_auto_handle": "no" if esc else "yes",
        "routing_reason": reason,
        "top1_similarity": ev[0]["score"] if ev else 0.0,
        "top5_support_examples": " || ".join(e["support_reply"] for e in ev),
    })
pred = pd.DataFrame(rows)
Path(c["paths"]["predictions"]).parent.mkdir(exist_ok=True)
pred.to_csv(c["paths"]["predictions"], index=False)
majority = g.gold_intent.value_counts().idxmax()
majority_pred = [majority] * len(g)
keyword = []
# A deliberately simple lexical baseline: choose the label whose name/description has the most token overlap.
from src.config import load_taxonomy
from src.intent import normalize
import re
tax = load_taxonomy()
for text in g.customer_text:
    toks = set(normalize(text).split())
    scores = {label: len(toks & set(normalize(label + " " + desc).split())) for label, desc in tax.items()}
    keyword.append(max(scores, key=scores.get))

result = {
    "n": len(g),
    "baseline_majority_accuracy": float(accuracy_score(g.gold_intent, majority_pred)),
    "baseline_majority_macro_f1": float(f1_score(g.gold_intent, majority_pred, average="macro", zero_division=0)),
    "baseline_lexical_accuracy": float(accuracy_score(g.gold_intent, keyword)),
    "baseline_lexical_macro_f1": float(f1_score(g.gold_intent, keyword, average="macro", zero_division=0)),
    "proposed_intent_accuracy": float(accuracy_score(g.gold_intent, pred.pred_intent)),
    "proposed_intent_macro_f1": float(f1_score(g.gold_intent, pred.pred_intent, average="macro", zero_division=0)),
    "proposed_routing_accuracy": float(accuracy_score(g.gold_auto_handle, pred.pred_auto_handle)),
    "retrieval_mean_top1_similarity": float(np.mean(pred.top1_similarity)),
}
json.dump(result, open("reports/evaluation.json", "w"), indent=2)
print(json.dumps(result, indent=2))
