import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .intent import normalize

class TfidfSupportIndex:
    def __init__(self,texts,replies,ids):
        self.vectorizer=TfidfVectorizer(preprocessor=normalize,ngram_range=(1,2),min_df=2,max_features=100000,sublinear_tf=True)
        self.matrix=self.vectorizer.fit_transform(texts)
        self.texts=list(texts); self.replies=list(replies); self.ids=list(ids)
    def search(self,query,k=5):
        scores=cosine_similarity(self.vectorizer.transform([query]),self.matrix)[0]
        idx=np.argsort(-scores)[:k]
        return [{"id":self.ids[i],"customer_text":self.texts[i],"support_reply":self.replies[i],"score":float(scores[i])} for i in idx]
    def save(self,path): joblib.dump(self,path)
    @staticmethod
    def load(path): return joblib.load(path)
