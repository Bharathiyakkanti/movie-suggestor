import streamlit as st
from recommender import (
    user_based_recommend,
    item_based_recommend,
    content_based_recommend,
    movies,
    user_movie_matrix
)

st.title("🎬 Movie Recommendation System")

menu = st.sidebar.selectbox(
    "Choose Recommendation Type",
    ["User-Based", "Item-Based", "Content-Based"]
)

if menu == "User-Based":
    user_id = st.selectbox("Select User ID", user_movie_matrix.index)
    if st.button("Recommend"):
        recs = user_based_recommend(user_id)
        st.write("### Recommended Movies")
        st.write(recs)

elif menu == "Item-Based":
    movie = st.selectbox("Select Movie", user_movie_matrix.columns)
    if st.button("Recommend"):
        recs = item_based_recommend(movie)
        st.write("### Similar Movies")
        st.write(recs)

elif menu == "Content-Based":
    movie = st.selectbox("Select Movie", movies['title'])
    if st.button("Recommend"):
        recs = content_based_recommend(movie)
        st.write("### Similar Movies")
        st.write(list(recs))
 