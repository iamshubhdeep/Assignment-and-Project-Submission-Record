import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
movies["genres"] = movies["genres"].fillna("")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()


def recommend(movie_title):
    if movie_title not in indices:
        return []

    idx = indices[movie_title]

    # Compute similarity only for the selected movie
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    movie_indices = sim_scores.argsort()[-6:-1][::-1]

    return movies["title"].iloc[movie_indices].tolist()


def get_movie_titles():
    return sorted(movies["title"].tolist())