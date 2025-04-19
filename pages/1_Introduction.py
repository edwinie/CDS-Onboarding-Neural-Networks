import streamlit as st

st.title("Project Introduction")
st.markdown("""
<p><a href="https://www.kaggle.com/datasets/anandshaw2001/top-spotify-songs-in-73-countries?resource=download" >Dataset</a></p>
""", unsafe_allow_html=True)


st.write("This project analyzes the top Spotify songs across 73 countries to uncover patterns and trends that contribute to a song’s popularity. "
"The dataset includes features such as danceability, energy, valence, tempo, acousticness, instrumentalness, liveness, speechiness, and loudness, providing a comprehensive view of musical characteristics. "
"We chose this dataset because it allows us to explore the relationship between audio features and song success on a global scale. Our primary goal is to predict a song's popularity based on its attributes and "
"identify regional differences in listener preferences through regression modeling and data visualization.")

