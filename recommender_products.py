# recommender_products.py - TITLE-ONLY (matches your CSV perfectly)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class ProductRecommender:
    def __init__(self):
        self.products = pd.read_csv('data/products.csv').head(2000)  # Limit for speed
        self.products = self.products.reset_index(drop=True)
        self.products['features'] = self.products['title'].fillna('Unknown product').str.lower()
        
        # TF-IDF on titles only
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.products['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Mappings
        self.id_to_idx = {asin: i for i, asin in enumerate(self.products['asin'])}
        self.idx_to_id = {i: asin for asin, i in self.id_to_idx.items()}
        self.idx_to_title = {i: title for i, title in enumerate(self.products['title'])}
        
        self.popular_products = self.products['asin'].value_counts().head(20).index.tolist()
    
    def get_recommendations(self, asin, top_k=10, conf_threshold=0.1):
        if asin not in self.id_to_idx:
            return [], 0.0, True
        idx = self.id_to_idx[asin]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        rec_ids = [self.idx_to_id[i[0]] for i in sim_scores]
        confidence = np.mean([self.cosine_sim[idx][self.id_to_idx[rid]] for rid in rec_ids])
        
        if confidence < conf_threshold:
            return self.popular_products[:top_k], confidence, True
        return rec_ids, confidence, False

# Cache it
if __name__ == "__main__":
    prod_rec = ProductRecommender()
    with open('data/prod_rec.pkl', 'wb') as f:
        pickle.dump(prod_rec, f)
    print("✅ Products recommender ready!")


