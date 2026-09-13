import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import load_config
from src.intent import normalize

p = argparse.ArgumentParser()
p.add_argument("--input", default=None)
p.add_argument("--k", type=int, default=12)
a = p.parse_args()
c = load_config()
path = a.input or c["paths"]["pairs_csv"]
df = pd.read_csv(path).dropna(subset=["customer_text"]).drop_duplicates("customer_text")
# Cap clustering input for fast reproducibility while preserving deterministic sampling.
df = df.sample(min(len(df), 30000), random_state=42).reset_index(drop=True)
vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
X = vec.fit_transform(df.customer_text)
k = min(a.k, len(df))
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(X)
terms = vec.get_feature_names_out()
rows = []
for cluster in range(k):
    mask = df.cluster == cluster
    center = km.cluster_centers_[cluster]
    top = center.argsort()[::-1][:12]
    examples = df.loc[mask, "customer_text"].head(5).tolist()
    rows.append({"cluster": cluster, "size": int(mask.sum()), "keywords": ", ".join(terms[top]), "examples": " || ".join(examples)})
out = pd.DataFrame(rows).sort_values("size", ascending=False)
out.to_csv("data/processed/intent_clusters.csv", index=False)
print(out.to_string(index=False))
