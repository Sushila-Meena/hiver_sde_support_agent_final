import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.intent import load,predict
from src.retrieval import TfidfSupportIndex
from src.routing import should_escalate
from src.agent import draft_reply
p=argparse.ArgumentParser(); p.add_argument("--text",required=True); a=p.parse_args()
c=load_config(); m=load(c["paths"]["model"]); idx=TfidfSupportIndex.load("data/processed/retrieval_index.joblib")
intent,conf,_=predict(m,a.text); ev=idx.search(a.text,c["model"]["retrieval_top_k"])
esc,reason=should_escalate(a.text,conf,ev,c["model"]["confidence_threshold"],c["routing"]["high_risk_patterns"])
draft=draft_reply(a.text,intent,ev)
print(json.dumps({"brand":c["brand"]["name"],"intent":intent,"confidence":round(conf,4),"decision":"HUMAN" if esc else "AUTO","reason":reason,"reply":draft["reply"],"evidence":ev},indent=2,ensure_ascii=False))
