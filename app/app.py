import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
.main-header {
    padding: 8px 0 20px 0;
    border-bottom: 2px solid #D4AF37;
    margin-bottom: 24px;
}
.main-header h1 { margin: 0; font-size: 30px; color: #F5F5F8; }
.main-header p { color: #9A9AA5; margin: 6px 0 0 0; font-size: 14px; }
.movie-card {
    background-color: #17181E;
    border: 1px solid #2C2D35;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.movie-title { color: #F0F0F2; font-weight: 600; font-size: 15px; }
.movie-genre { color: #9A9AA5; font-size: 12px; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎬 CineMatch</h1>
    <p>Hybrid recommendations — content similarity meets real viewing behavior</p>
</div>
""", unsafe_allow_html=True)

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
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:n]
    movie_indices = [i[0] for i in hybrid_scores]
    return movies[['title', 'genres']].iloc[movie_indices]

# ---------- UI ----------
min_id = int(ratings['userId'].min())
max_id = int(ratings['userId'].max())
user_id = st.number_input("Enter User ID", min_value=min_id, max_value=max_id, value=min_id)
movie_title = st.selectbox("Select a movie you like", movies['title'].values)

if st.button("✨ Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        results = hybrid_recommendations(user_id, movie_title)
    st.markdown("### Recommended for you")
    for idx, row in results.iterrows():
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{row['title']}</div>
            <div class="movie-genre">{row['genres']}</div>
        </div>
        """, unsafe_allow_html=True)