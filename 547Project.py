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
custom_stopwords = STOPWORDS.union({'concert', 'show', 'see', 'one', 'Billie', 'Beyonce', 'Beyoncé', 'Madonna', 'Katy', 'Perry', 'Taylor', 'Swift'})

def generate_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 400, background_color = 'white', stopwords = custom_stopwords).generate(text)
    plt.figure(figsize = (10, 5))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    plt.show()



# ---------------------------
# OpenAI API - Review
# ---------------------------
def provide_feedback(review_text, openai_api_key):
    client = OpenAI(api_key = openai_api_key)
    
    prompt = "Analyze the sentiment of the following review and suggest an appropriate response:\nReview: " + review_text

    chat_completion = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 100,
        temperature = 0.7
    )
   
    return chat_completion.choices[0].message.content



# ---------------------------
# OpenAI API - Improvement
# ---------------------------
def analyze_sentiment(review_improvement, openai_api_key):
    client = OpenAI(api_key = openai_api_key)
    
    combined_reviews = " ".join(review_improvement[0:77])
    prompt = "You are a concert organizer. Based on the following reviews of concerts, please highlight top 5 areas for improvements.\n" + combined_reviews

    chat_completion = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 250,
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
artist_reviews = df[df['Artist'] == artist_selection]['Review']
artist_data = df[df['Artist'] == artist_selection]

# positive & negative feedback
positive_feedback = artist_data[artist_data['Rating'] > 3]['Review']
negative_feedback = artist_data[artist_data['Rating'] <= 3]['Review']

# feedbacks for future improvement
review_improvement = artist_data[artist_data['Rating'] <= 2]['Review']




# ----------------------------------
# Streamlit - Main - Word Cloud
# ----------------------------------
st.subheader('Word Cloud')

col1, col2 = st.columns(2)

# positive word cloud
with col1:
    st.write("Positive Keywords")
    combined_text = " ".join(positive_feedback)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
    generate_wordcloud(combined_text)
    st.pyplot()

# negative word cloud
with col2:
    st.write("Negative Keywords")
    combined_text = " ".join(negative_feedback)
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
    generate_wordcloud(combined_text)
    st.pyplot()



# ----------------------------------
# Streamlit - Main - Improvement
# ----------------------------------
st.write('\n\n')
st.subheader('Potential Improvements')

# Button to run analysis
if st.button("Analyze Potential Improvements"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key to proceed.")
    else:
        feedback = analyze_sentiment(review_improvement, openai_api_key)
        st.success(feedback)




# ----------------------------------
# Streamlit - Main - Chatbot
# ----------------------------------
st.write('\n\n')
st.subheader('Feedback')

# Textbox for user to enter their review
review = st.text_area("Please write your review here:")

# Button to submit review
if st.button("Submit Review"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key to proceed.")
    elif not review:
        st.error("Please write a review to submit.")
    else:
        # Analyze the review's sentiment and provide feedback
        feedback = provide_feedback(review, openai_api_key)
        st.success(feedback)
