import streamlit as st
import pandas as pd



# Load the dataset
df = pd.read_csv('https://github.com/xydui/547Project/blob/main/Combined.csv')
print(df.head())




# Streamlit Layout
st.title('Concert Feedback Analysis')

artist_selection = st.selectbox('Select an Artist', df['Artist'].unique())
artist_reviews = df[df['Artist'] == artist_selection]['Review']



st.write(artist_reviews[0:3])
