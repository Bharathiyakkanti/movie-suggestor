import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Sample top 1000 users to reduce memory
top_users = ratings['userId'].value_counts().head(1000).index
ratings = ratings[ratings['userId'].isin(top_users)]

# Sample top 2000 movies to reduce memory
top_movies = ratings['movieId'].value_counts().head(2000).index
ratings = ratings[ratings['movieId'].isin(top_movies)]

# Merge data
data = pd.merge(ratings, movies, on="movieId")

# Create User-Item Matrix using pivot, but with sampled data
user_movie_matrix = data.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

user_movie_filled = user_movie_matrix.fillna(0)

# USER-BASED SIMILARITY
user_similarity = cosine_similarity(user_movie_filled)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_filled.index,
    columns=user_movie_filled.index
)

def user_based_recommend(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return pd.Series(dtype=float)

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_scores = user_movie_filled.loc[similar_users.index].T.dot(similar_users)
    recommendations = weighted_scores / similar_users.sum()
    watched = user_movie_matrix.loc[user_id]
    recommendations = recommendations[watched.isna()]
    return recommendations.sort_values(ascending=False).head(top_n)

# ITEM-BASED SIMILARITY - commented out due to memory constraints
item_similarity = cosine_similarity(user_movie_filled.T)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_movie_filled.columns,
    columns=user_movie_filled.columns
)

def item_based_recommend(movie_title, top_n=5):
    return item_similarity_df[movie_title].sort_values(ascending=False)[1:top_n+1]

# CONTENT-BASED FILTERING
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('').replace('(no genres listed)', '')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

def content_based_recommend(movie_title, top_n=5):
    if movie_title not in movies['title'].values:
        return pd.Series(dtype=object)

    idx = movies[movies['title'] == movie_title].index[0]
    movie_vector = tfidf_matrix[idx]
    content_similarity = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    scores = list(enumerate(content_similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return movies.iloc[movie_indices]['title']
