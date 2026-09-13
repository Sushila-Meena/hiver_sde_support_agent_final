import re, joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def normalize(text):
    text=str(text).lower()
    text=re.sub(r"https?://\S+"," URL ",text)
    text=re.sub(r"@\S+"," USER ",text)
    return re.sub(r"\s+"," ",text).strip()

def build_classifier():
    return Pipeline([
        ("tfidf",TfidfVectorizer(preprocessor=normalize,ngram_range=(1,2),min_df=2,max_df=.98,sublinear_tf=True,max_features=80000)),
        ("clf",LogisticRegression(C=3,max_iter=1000,class_weight="balanced"))
    ])

def train_classifier(texts,labels):
    m=build_classifier(); m.fit(texts,labels); return m

def predict(model,text):
    p=model.predict_proba([text])[0]; i=int(np.argmax(p))
    return model.classes_[i],float(p[i]),dict(zip(model.classes_,map(float,p)))

def save(model,path): joblib.dump(model,path)
def load(path): return joblib.load(path)
