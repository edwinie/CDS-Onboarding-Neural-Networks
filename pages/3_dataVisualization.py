import streamlit as st

st.title("Data Visualization")

st.write("Our most important data vizualization would be our correlation matrix because we can actually see how the different factors are actually affecting song population, which is the point of our project. We also created scatterplots to visualize the correlation to see how strong it is.  ")

st.image("correlation_matrix_popularity.png", caption="Correlation Matrix")

st.image("scatterplot_energy_loudness.png", caption="Energy & Loudness Correlation")