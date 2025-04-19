import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Manipulation")

st.write("We decided to clean all the missing values out of the dataset and replace all the NaN values with a summary statistic. We did it this way because it made for not losing a lot of data so our dataset would be more complete. ")

st.subheader("Cleaning Strategy")
st.markdown("""
- Removed columns with more than 50% missing values.
- Filled numerical NaNs with the column median.
- Filled categorical NaNs with the mode (most frequent value).
""")



