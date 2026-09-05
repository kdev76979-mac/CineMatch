import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_models():
    with open('models/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('models/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    movies = pd.read_pickle('models/movies_df.pkl')
    ratings = pd.read_pickle('models/ratings_df.pkl')
    
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return tfidf_matrix, cosine_sim, svd_model, movies, ratings

tfidf_matrix, cosine_sim, svd_model, movies, ratings = load_models()

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def hybrid_recommendations(user_id, title, n=10, content_weight=0.5, collab_weight=0.5):
    idx = indices[title]
    input_movie_id = movies.iloc[idx]['movieId']
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    hybrid_scores = []
    for i, content_score in sim_scores:
        movie_id = movies.iloc[i]['movieId']
        if movie_id == input_movie_id:
            continue
        collab_score = svd_model.predict(user_id, movie_id).est / 5.0
        combined = (content_weight * content_score) + (collab_weight * collab_score)
        hybrid_scores.append((i, combined))
    
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    hybrid_scores = hybrid_scores[:n]
    
    movie_indices = [i[0] for i in hybrid_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# ---------- UI ----------
st.title("🎬 CineMatch — Movie Recommender")
st.write("Hybrid recommendation system (Content-Based + Collaborative Filtering)")

min_id = int(ratings['userId'].min())
max_id = int(ratings['userId'].max())
user_id = st.number_input("Enter User ID", min_value=min_id, max_value=max_id, value=min_id)
movie_title = st.selectbox("Select a movie you like", movies['title'].values)

if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        results = hybrid_recommendations(user_id, movie_title)
    st.subheader("Recommended Movies:")
    for idx, row in results.iterrows():
        st.write(f"**{row['title']}** — {row['genres']}")