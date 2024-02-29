import streamlit as st
import pandas as pd
import openai
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv('https://raw.githubusercontent.com/xydui/547Project/main/Combined.csv')
print(df.head())


# ---------------------------
# Word Cloud Settings
# ---------------------------
custom_stopwords = STOPWORDS.union({'concert', 'show', 'see', 'one'})

def generate_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white', stopwords = custom_stopwords).generate(text)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# ---------------------------
# Streamlit Layout
# ---------------------------
# title
st.title('Concert Feedback Analysis')

# Textbox for entering OpenAI API key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Save the API key in session state to reuse it for the session
if api_key:
    st.session_state['api_key'] = api_key

# get data from dataset
artist_selection = st.selectbox('Select an Artist', df['Artist'].unique())
artist_reviews = df[df['Artist'] == artist_selection]['Review'].tolist()

# word cloud
if st.button('Show Word Cloud'):
    combined_text = " ".join(artist_reviews)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
    generate_wordcloud(combined_text)
    st.pyplot()



# Textbox for user to enter their review
review = st.text_area("Write your review here:")

# Function to analyze sentiment
def analyze_sentiment(review_text, api_key):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{review_text}"}
        ]
    )
    return response['choices'][0]['message']['content']



# Button to submit review
if st.button("Submit Review"):
    if not api_key:
        st.error("Please enter your OpenAI API key to proceed.")
    elif not review:
        st.error("Please write a review to submit.")
    else:
        # Analyze the review's sentiment and provide feedback
        feedback = analyze_sentiment(review, st.session_state['api_key'])
        st.success(feedback)
