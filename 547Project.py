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
def generate_wordcloud(custom_stopwords, text):
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
    
    prompt = "You are a concert orgainzer. Read the following concert review and suggest an appropriate response:\nReview: " + review_text

    chat_completion = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 200,
        temperature = 0.7
    )
  
    return chat_completion.choices[0].message.content



# ---------------------------
# OpenAI API - Recommendation
# ---------------------------
def provide_recommendation(artist):
    client = OpenAI(api_key = openai_api_key)

    prompt = "You are a member of the artist management company. Please recommend to the fans of " + artist + \
             " some recent albums, movies, books or other works related to the artist. " + \
             "Please make the response in bullet points and sound like a member of artist management team. " + \
             "Please remove the beginning like 'certainly' or 'dear fans'."

    chat_completion = client.chat.completions.create(
        model = "gpt-4-0125-preview",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 500,
        temperature = 0.7
    )

    return chat_completion.choices[0].message.content



# ---------------------------
# OpenAI API - Improvement
# ---------------------------
def analyze_sentiment(review_improvement, openai_api_key):
    client = OpenAI(api_key = openai_api_key)
    
    combined_reviews = " ".join(review_improvement[0:80])
    prompt = "You are a concert organizer. Based on the following reviews of concerts, please highlight top 5 areas for improvements.\n" + combined_reviews

    chat_completion = client.chat.completions.create(
        model = "gpt-3.5-turbo-0125",
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 400,
        temperature = 0.7
    )
   
    return chat_completion.choices[0].message.content



# ---------------------------
# Streamlit - Config
# ---------------------------
# page settings
st.set_page_config(
    page_title = "Review I.Q.",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

# title
st.title('Review I.Q.')

# tabbed navigation
CustomerTab, OrganizerTab = st.tabs(['Concertgoers', 'Organizers'])



# ---------------------------
# Streamlit - Sidebar
# ---------------------------
# logo
st.sidebar.image('https://raw.githubusercontent.com/xydui/547Project/main/logo.png')

# Artist Selection
artist_selection = st.sidebar.selectbox('Select an Artist:', df['Artist'].unique())

# Textbox for entering OpenAI API key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type = "password")

# info about the app
st.sidebar.markdown('-----')
st.sidebar.write('Contact us on [Review I.Q.](www.Review-IQ.com).')



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
# Streamlit - Customer Tab
# ----------------------------------
with CustomerTab:

    # ------------ Review Chatbot ---------- #
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
            feedback = provide_feedback(review, openai_api_key)
            st.success(feedback)
    


    # ------------ Recommendation ---------- #
    st.subheader('Recommendations')

    # button to get recommendations
    if st.button("Get Recommendations"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key to proceed.")
        else:
            recommendation = provide_recommendation(artist_selection)
            st.success(recommendation)




# ----------------------------------
# Streamlit - Orgainzer Tab
# ----------------------------------
with OrganizerTab:

    # ------------ Word Cloud ---------- #
    st.subheader('Word Cloud')
    col1, col2 = st.columns(2)

    # positive word cloud
    positive_stopwords = STOPWORDS.union({'concert', 'show', 'see', 'one', 'Billie', 'Beyonce', 'Beyoncé', 'Madonna', 'Katy', 'Perry', 'Taylor', 'Swift'})
    
    with col1:
        st.write("Positive Keywords")
        combined_text = " ".join(positive_feedback)
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
        generate_wordcloud(positive_stopwords, combined_text)
        st.pyplot()

    # negative word cloud
    negative_stopwords = STOPWORDS.union({'concert', 'show', 'see', 'one', 'Billie', 'Beyonce', 'Beyoncé', 'Madonna', 'Katy', 'Perry', 'Taylor', 'Swift', 'great', 'good', 'amazing', 'love'})

    with col2:
        st.write("Negative Keywords")
        combined_text = " ".join(negative_feedback)
        st.set_option('deprecation.showPyplotGlobalUse', False)  # To hide warning
        generate_wordcloud(negative_stopwords, combined_text)
        st.pyplot()
    


    # ------------ Improvement ---------- #
    st.write('\n\n')
    st.subheader('Areas for Improvements')

    # Button to run analysis
    if st.button("Analyze Potential Improvements"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key to proceed.")
        else:
            feedback = analyze_sentiment(review_improvement, openai_api_key)
            st.success(feedback)
