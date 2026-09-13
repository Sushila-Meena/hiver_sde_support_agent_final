import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error
from src.evaluation import pearson_or_none

g = pd.read_csv("data/golden/judge_calibration.csv")
g["human_reply_quality"] = pd.to_numeric(g["human_reply_quality"], errors="coerce")
g = g.dropna(subset=["human_reply_quality", "judge_overall"]).copy()
if len(g) < 2:
    raise SystemExit("Need at least 2 human-labelled calibration rows.")
h = g.human_reply_quality.astype(int)
j = g.judge_overall.astype(int)
result = {
    "n": len(g),
    "pearson_correlation": pearson_or_none(h, j),
    "mean_absolute_error": float(mean_absolute_error(h, j)),
    "weighted_kappa": float(cohen_kappa_score(h, j, weights="quadratic")),
}
Path("reports").mkdir(exist_ok=True)
json.dump(result, open("reports/judge_calibration.json", "w"), indent=2)
print(json.dumps(result, indent=2))
