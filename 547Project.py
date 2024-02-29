import streamlit as st
import pandas as pd



# Load the dataset
df = pd.read_excel('https://raw.githubusercontent.com/xydui/547Project/blob/5ff6d690d7481782fefe62d942340b7755cfc726/Combined.xlsx')
print(df.head())




# Streamlit Layout
st.title('Concert Feedback Analysis')

artist_selection = st.selectbox('Select an Artist', df['Artist'].unique())
artist_reviews = df[df['Artist'] == artist_selection]['Review']



st.write(artist_reviews[0:3])
