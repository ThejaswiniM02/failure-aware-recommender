import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class MovieRecommender:
    def __init__(self):
        # Load MovieLens movies.csv
        self.movies = pd.read_csv('data/movies.csv')
        self.movies = self.movies.reset_index(drop=True)
        
        # Create features: title + genres
        self.movies['features'] = (
            self.movies['title'].fillna('') + ' ' + 
            self.movies['genres'].fillna('')
        )
        
        # TF-IDF vectorization
        self.tfidf = TfidfVectorizer(
            stop_words='english', 
            max_features=5000, 
            min_df=2
        )
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Cache precomputed popular movies for fallback
        self.popular_movies = self.movies.index.value_counts().head(20).tolist()
    
    def recommend(self, item_index, top_k=10):
        """Get top-K similar movies by index"""
        sim_scores = list(enumerate(self.cosine_sim[item_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        return [i[0] for i in sim_scores]
    
    def get_recommendations(self, item_index, top_k=10, conf_threshold=0.1):
        """
        Main function for app.py:
        Returns (recommended_indices, confidence_score, fallback_used)
        """
        rec_indices = self.recommend(item_index, top_k)
        sim_scores = self.cosine_sim[item_index][rec_indices]
        confidence = np.mean(sim_scores)
        
        # Low confidence? Use fallback
        if confidence < conf_threshold:
            return self.popular_movies[:top_k], confidence, True
        
        return rec_indices, confidence, False

# Global instance (loaded once)
if os.path.exists('data/movie_rec.pkl'):
    with open('data/movie_rec.pkl', 'rb') as f:
        rec = pickle.load(f)
else:
    rec = MovieRecommender()
    os.makedirs('data', exist_ok=True)
    with open('data/movie_rec.pkl', 'wb') as f:
        pickle.dump(rec, f)

# Export for app.py
get_recommendations = rec.get_recommendations
