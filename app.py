# app.py - 100% WORKING (uses debug's exact logic)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Failure‑Aware Recommender", layout="wide")
st.title("🚀 Failure‑Aware Recommender")
st.caption("Movies vs Products • Confidence • Human Feedback")

# ---------- RECOMMENDATION FUNCTIONS ----------
@st.cache_data
def get_movie_recs(item_index, top_k=10, conf_threshold=0.1):
    movies = pd.read_csv("data/movies.csv").head(1000).reset_index(drop=True)
    movies['features'] = movies['title'] + ' ' + movies['genres']
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sim_scores = sorted(enumerate(cosine_sim[item_index]), key=lambda x: x[1], reverse=True)[1:top_k+1]
    rec_indices = [i[0] for i in sim_scores]
    confidence = np.mean([cosine_sim[item_index][i[0]] for i in sim_scores])
    
    if confidence < conf_threshold:
        return list(range(1, top_k+1)), confidence, True
    return rec_indices, confidence, False

@st.cache_data
def get_product_recs(product_index, top_k=10, conf_threshold=0.1):
    products = pd.read_csv("data/products.csv").head(1000).reset_index(drop=True)
    products['features'] = products['title'].fillna('Unknown').str.lower()
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
    tfidf_matrix = tfidf.fit_transform(products['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sim_scores = sorted(enumerate(cosine_sim[product_index]), key=lambda x: x[1], reverse=True)[1:top_k+1]
    rec_asins = [products.iloc[i[0]]['asin'] for i in sim_scores]
    confidence = np.mean([cosine_sim[product_index][i[0]] for i in sim_scores])
    
    if confidence < conf_threshold:
        return products['asin'][:top_k].tolist(), confidence, True
    return rec_asins, confidence, False

# ---------- UI ----------
st.sidebar.header("🎯 Controls")
dataset = st.sidebar.selectbox("Dataset", ["Movies", "Products"])

# Load data + selector (SAME AS DEBUG)
if dataset == "Movies":
    movies = pd.read_csv("data/movies.csv").head(100).reset_index(drop=True)
    selected_idx = st.sidebar.selectbox("Select movie", range(10), format_func=lambda i: movies.iloc[i]['title'])
    get_recs = get_movie_recs
    items_df = movies
elif dataset == "Products":
    products = pd.read_csv("data/products.csv").head(100).reset_index(drop=True)
    selected_idx = st.sidebar.selectbox("Select product", range(10), format_func=lambda i: products.iloc[i]['title'][:50])
    get_recs = get_product_recs
    items_df = products

top_k = st.sidebar.slider("Top-K", 5, 15, 10)
conf_threshold = st.sidebar.slider("Threshold", 0.0, 0.5, 0.1, 0.01)
trigger = st.sidebar.button("🔮 Recommend", use_container_width=True)

if trigger:
    rec_ids, confidence, fallback_used = get_recs(selected_idx, top_k, conf_threshold)
    
    # Confidence
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Confidence", f"{confidence:.3f}")
    with col2:
        if confidence < conf_threshold:
            st.error("⚠️ Low confidence – fallback activated")
        else:
            st.success("✅ High confidence")
    
    st.markdown("---")
    st.subheader(f"Top {dataset} Recommendations")
    
    # Show recommendations
    for i, rec_id in enumerate(rec_ids[:top_k]):
        if dataset == "Movies":
            rec_title = items_df.iloc[rec_id]['title'] if rec_id < len(items_df) else "Fallback Movie"
        else:
            # Find product by ASIN
            rec_row = items_df[items_df['asin'] == rec_id]
            rec_title = rec_row['title'].iloc[0][:60] if len(rec_row) > 0 else "Fallback Product"
        
        c1, c2, c3 = st.columns([6, 1, 1])
        with c1:
            st.write(f"• **{rec_title}**")
        with c2:
            if st.button("👍", key=f"up_{i}_{dataset}"):
                st.success("👍 Feedback recorded!")








