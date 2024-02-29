import streamlit as st
import pandas as pd
from openai import OpenAI
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv('https://raw.githubusercontent.com/xydui/547Project/main/Combined.csv')



# ---------------------------
# Word Cloud Settings
# ---------------------------
custom_stopwords = STOPWORDS.union({'concert', 'show', 'see', 'one', 'Billie', 'Beyonce', 'Madonna', 'Katy', 'Perry', 'Taylor', 'Swift'})

def generate_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white', stopwords = custom_stopwords).generate(text)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.show()



# ---------------------------
# OpenAI API
# ---------------------------
def analyze_sentiment(review_text, openai_api_key):
    client = OpenAI(api_key = openai_api_key)
    
    prompt = "Analyze the sentiment of the following review and suggest an appropriate response:\nReview: " + review_text

    chat_completion = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 50,
        temperature = 0.7
    )
   
    return chat_completion.choices[0].message.content




# ---------------------------
# Streamlit - Config
# ---------------------------
# page settings
st.set_page_config(
    page_title = "Concert Feedback Analysis",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# title
st.title('🎸 Concert Feedback Analysis')



# ---------------------------
# Streamlit - Sidebar
# ---------------------------
# Artist Selection
artist_selection = st.sidebar.selectbox('Select an Artist', df['Artist'].unique())

# Textbox for entering OpenAI API key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type = "password")



# ----------------------------------
# Streamlit - Main
# ----------------------------------
# get data from dataset
artist_reviews = df[df['Artist'] == artist_selection]['Review'].tolist()



# ----------------------------------
# Streamlit - Main - Word Cloud
# ----------------------------------
st.subheader('Word Cloud')

col1, col2 = st.columns(2)

# positive word cloud
with col1:
    st.write("Positive Keywords")
    combined_text = " ".join(artist_reviews)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
    generate_wordcloud(combined_text)
    st.pyplot()

# negative word cloud
with col2:
    st.write("Positive Keywords")
    combined_text = " ".join(artist_reviews)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
    generate_wordcloud(combined_text)
    st.pyplot()



# ----------------------------------
# Streamlit - Main - Chatbot
# ----------------------------------
st.write("\n\n")
st.subheader('Feedback')

# Textbox for user to enter their review
review = st.text_area("Write your review here:")

# Button to submit review
if st.button("Submit Review"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key to proceed.")
    elif not review:
        st.error("Please write a review to submit.")
    else:
        # Analyze the review's sentiment and provide feedback
        feedback = analyze_sentiment(review, openai_api_key)
        st.success(feedback)
