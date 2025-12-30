# benchmark.py - NO FORMAT ERRORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("🚀 Benchmarking MovieLens vs Amazon Products...")

def benchmark_movies(n_samples=50):
    movies = pd.read_csv('data/movies.csv').head(1000).reset_index(drop=True)
    movies['features'] = movies['title'] + ' ' + movies['genres']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    confidences = []
    fallbacks = 0
    
    for i in range(min(n_samples, len(movies))):
        sim_scores = sorted(enumerate(cosine_sim[i]), key=lambda x: x[1], reverse=True)[1:11]
        confidence = np.mean([x[1] for x in sim_scores])
        if confidence < 0.1:
            fallbacks += 1
        confidences.append(confidence)
    
    return np.mean(confidences), fallbacks / n_samples

def benchmark_products(n_samples=50):
    products = pd.read_csv('data/products.csv').head(1000).reset_index(drop=True)
    products['features'] = products['title'].fillna('Unknown').str.lower()
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(products['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    confidences = []
    fallbacks = 0
    
    for i in range(min(n_samples, len(products))):
        sim_scores = sorted(enumerate(cosine_sim[i]), key=lambda x: x[1], reverse=True)[1:11]
        confidence = np.mean([x[1] for x in sim_scores])
        if confidence < 0.1:
            fallbacks += 1
        confidences.append(confidence)
    
    return np.mean(confidences), fallbacks / n_samples

# Run
print("Movies...")
movie_conf, movie_fallback = benchmark_movies()
print("Products...")
prod_conf, prod_fallback = benchmark_products()

# Results (SIMPLE FORMATTING)
print("\n" + "="*50)
print("📊 PERFORMANCE COMPARISON")
print("="*50)
print("Avg Confidence:    Movies =", f"{movie_conf:.3f}", "Products =", f"{prod_conf:.3f}")
print("Fallback Rate:     Movies =", f"{movie_fallback:.1%}", "Products =", f"{prod_fallback:.1%}")

# Save for README
with open('benchmark_results.txt', 'w') as f:
    f.write(f"| Avg Confidence | {movie_conf:.3f} | {prod_conf:.3f} | Movies |\n")
    f.write(f"| Fallback Rate  | {movie_fallback:.1%} | {prod_fallback:.1%} | Movies |\n")

print("\n✅ Results saved to benchmark_results.txt")
print("\n📋 Copy this to README:")
print(f"| Avg Confidence | {movie_conf:.3f} | {prod_conf:.3f} | Movies |")
print(f"| Fallback Rate  | {movie_fallback:.1%} | {prod_fallback:.1%} | Movies |")

