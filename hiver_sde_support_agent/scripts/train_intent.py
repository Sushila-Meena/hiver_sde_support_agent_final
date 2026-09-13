import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.config import load_config
from src.intent import train_classifier, save

c = load_config()
df = pd.read_csv(c["paths"]["train_labels"])
df["intent"] = df["intent"].fillna("").str.strip()
df = df[df.intent.ne("")].copy()
counts = df.intent.value_counts()
df = df[df.intent.isin(counts[counts >= c["model"]["min_train_per_intent"]].index)]
if df["intent"].nunique() < 2:
    raise ValueError("Need at least 2 manually labelled intents before training.")
tr, te = train_test_split(df, test_size=.20, random_state=42, stratify=df.intent)
model = train_classifier(tr.customer_text, tr.intent)
pred = model.predict(te.customer_text)
Path(c["paths"]["model"]).parent.mkdir(parents=True, exist_ok=True)
save(model, c["paths"]["model"])
report = classification_report(te.intent, pred, output_dict=True, zero_division=0)
Path("reports").mkdir(exist_ok=True)
json.dump({"n": len(te), "report": report}, open("reports/intent_validation.json", "w"), indent=2)
print(f"Saved model trained on {len(tr)} manually labelled examples.")
print(json.dumps({k: report[k] for k in ["accuracy", "macro avg"] if k in report}, indent=2))
