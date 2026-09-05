# 🎬 CineMatch — Hybrid Movie Recommendation System

A movie recommendation system combining **Content-Based Filtering** and **Collaborative Filtering** to suggest personalized movies.

## 🚀 Live Demo
https://cinematch-iritelyde3kreukymt8oku.streamlit.app

## 📌 Overview
This project builds a hybrid recommender using the MovieLens dataset. Instead of relying on a single approach, it combines:
- **Content-Based Filtering** — uses TF-IDF on movie genres + cosine similarity
- **Collaborative Filtering** — uses SVD (matrix factorization) on user ratings
- **Hybrid** — combines both scores to balance genre-similarity with actual user behavior

## 🧠 Why Hybrid?
The dataset has **~98% sparsity** (610 users, 9,724 movies, only 100K ratings) — meaning pure collaborative filtering struggles with cold-start scenarios. Content-based filtering compensates for this by using genre metadata, which is always available even for less-rated movies.

## 📊 Dataset
- MovieLens Latest Small Dataset (100,836 ratings, 9,742 movies, 610 users)
- Source: [GroupLens](https://grouplens.org/datasets/movielens/latest/)

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- scikit-learn (TF-IDF, Cosine Similarity)
- Surprise (SVD collaborative filtering)
- Streamlit (deployment)

## 📈 Model Performance
- SVD Collaborative Filtering RMSE: **0.88** (rating scale 0.5–5.0)

## ⚠️ Known Limitations
- Content-based similarity is slightly biased toward rare tags (e.g., "IMAX") due to TF-IDF's inverse document frequency weighting — movies sharing rare genre tags get inflated similarity scores.
- Cold-start problem still exists for brand-new users with zero rating history.
- Genre-only content features don't capture plot, cast, or director similarity.

## 🏃 How to Run Locally
```bash
git clone https://github.com/kdev76979-mac/CineMatch.git
cd CineMatch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Run notebooks/eda.ipynb first to generate models/ folder
streamlit run app/app.py
```

## 📂 Project Structure