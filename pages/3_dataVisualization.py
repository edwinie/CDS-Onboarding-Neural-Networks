import streamlit as st

st.title("Data Visualization")

st.write("Our most important data visualization would be our correlation matrix because we can actually see how the different factors are actually affecting song population, which is the point of our project. We also created scatterplots to visualize the correlation to see how strong it is.  ")

st.image("correlation_matrix_popularity.png", caption="Correlation Matrix")

st.write("The strongest positive correlation is between valence and danceability, with a value of 0.42. This makes sense because songs that are happier or more positive in tone tend to be more danceable.")

st.write("The strongest negative correlations are around -0.38 seen between instrumentalness and loudness, and loudness and acousticness. This tells us that louder songs are generally less instrumental and less acoustic because loudness often comes from more energetic tracks.")

st.write("These stronger relationships are not directly with popularity, but they reveal patterns in how certain audio features interact. It shows that while popularity itself doesn't strongly correlate with individual features, the dataset does contain meaningful structure between those features.")

st.image("scatterplot_energy_loudness.png", caption="Energy & Loudness Correlation")
