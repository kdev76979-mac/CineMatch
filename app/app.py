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