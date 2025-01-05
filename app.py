
import streamlit as st
import json
import random
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer

# Load NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model, words, and classes
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents_data = json.loads(open('intents.json').read())  # Renamed to avoid conflict

# Define chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."


st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;
        color: #000000;
    }
    .css-145kmo2 {
        background-color: #F5F5F5;
        border-radius: 10px;
        color: #000000;
    }
    .stTextInput>div>div {
        background-color: #F5F5F5;
        color: #000000;
    }
    .css-15zrgzn {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Home", "Conversation History", "About"])

# Main content based on the menu selection
if menu == "Home":
    st.title("🤖 Intents of Chatbot using NLP")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
    user_message = st.text_input("You:", key="input_message", placeholder="Type your message here...")
    if user_message:
        predicted_intents = predict_class(user_message)  # Renamed for clarity
        bot_response = get_response(predicted_intents, intents_data)
        st.text_area("Chatbot:", value=bot_response, height=100)

elif menu == "Conversation History":
    st.title("🕒 Conversation History")
    st.write("This section can show the saved conversation history in the future.")

elif menu == "About":
    st.title("ℹ️ About the Chatbot")
    st.write("This chatbot demonstrates the use of Natural Language Processing (NLP) with TensorFlow and Streamlit.")

st.sidebar.markdown("---")
st.sidebar.write("Developed by [Himanshu sahu].")
