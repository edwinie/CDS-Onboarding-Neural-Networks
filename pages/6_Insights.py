import streamlit as st

st.title("Project Insights")

st.write("KNN outperformed Decision Trees and SVR for Spotify popularity prediction because it naturally captures the local patterns in music popularity where similar songs tend to have " \
"similar popularity ratings. Unlike DT, which creates rigid decision boundaries, and SVR, which fits a global function, KNN's flexible neighborhood-based approach better handles the complex, " \
"non-linear relationships between audio features and popularity. The averaging effect of using " \
"k=10 neighbors also provided robustness against the noise and outliers that are common in music preference data")

st.image("streamlitImages/msecomparison.png")