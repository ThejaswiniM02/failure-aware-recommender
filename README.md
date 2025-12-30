🚀 Failure-Aware Recommender System

A content-based recommendation system that detects low-confidence predictions, applies safe fallback strategies, and incorporates human-in-the-loop feedback to improve future recommendations.

Built in ~5 hours using TF-IDF + cosine similarity and demonstrated via a lightweight Streamlit UI.

✨ Key Features

Failure Awareness
Computes a confidence score for each recommendation set and detects unreliable outputs.

Fallback Strategy
When confidence falls below a threshold, the system switches to popularity-based recommendations to avoid poor UX.

Human-in-the-Loop Feedback
Users can upvote or downvote recommendations, which dynamically re-ranks future results.

Live Demo (Streamlit)
UI is intentionally minimal and used only to simulate real-world feedback loops.

🧠 Why This Project

Most recommender demos stop at “here are similar items.”

This project focuses on what happens when recommendations fail:

Sparse metadata

Weak similarity neighborhoods

Cold-start–like scenarios

By explicitly modeling confidence and allowing human correction, the system mirrors how production recommendation pipelines are safeguarded in practice.

🏗️ System Design

Pipeline

Item text → TF-IDF vectorization

Cosine similarity for candidate retrieval

Confidence = mean similarity of top-K items

Low confidence → fallback strategy

Human feedback → score adjustment and re-ranking

📊 Evaluation Summary
Metric	Baseline Recommender	Failure-Aware System
Avg Confidence	Lower	Higher
Fallback Coverage	N/A	Enabled
User Correction	❌	✅

(Evaluation focuses on reliability and coverage rather than absolute accuracy.)

🧪 Technical Details

Vectorization: TF-IDF (English stopwords removed)

Similarity: Cosine similarity

Confidence Metric:
confidence = mean(top_k_similarity_scores)

Fallback Trigger:
confidence < threshold

Feedback Handling:
Positive feedback boosts similarity, negative feedback penalizes it

🖥️ Streamlit UI (Purposefully Minimal)

The UI exists only to:

Display confidence scores

Flag fallback usage

Capture human feedback (👍 / 👎)

No dashboards. No visual noise. The focus stays on ML behavior.

📁 Repository Structure
failure-aware-recommender/
│
├── app.py            # Streamlit UI
├── recommender.py    # Core recommendation + confidence logic
├── feedback.py       # Human feedback handling
├── data/             # Sample datasets
├── screenshots/      # Demo GIF / images
└── requirements.txt

⚡ Setup
pip install -r requirements.txt
streamlit run app.py

🔮 Future Improvements

Replace heuristic feedback weighting with online learning

Extend to user-based personalization

Add concept-drift monitoring

Evaluate ranking metrics (NDCG / MAP)