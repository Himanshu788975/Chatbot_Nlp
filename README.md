# Chatbot_Nlp
#  Chatbot Using NLP and Streamlit

This project is a chatbot application built using **Natural Language Processing (NLP)** and deployed with **Streamlit**. The chatbot predicts user intents and generates responses based on a pre-trained model.

##  Features

- Interactive chatbot interface
- Intent classification using a trained TensorFlow model
- Modular design for extensibility
- Sidebar navigation for additional features like conversation history and app information
- Custom styling for a better user experience

##  Technologies Used

- **Python**: Core programming language
- **TensorFlow**: Machine learning model for intent classification
- **Streamlit**: Web framework for building interactive user interfaces
- **NLTK**: Natural language processing toolkit for tokenization and lemmatization

##  How to Run the Project

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Required Python libraries (install via `requirements.txt`)

pip install -r requirements.txt

Download NLTK data:

python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"


Ensure you have the following files:

chatbot_model.h5: Trained TensorFlow model
words.pkl: Pickled words list
classes.pkl: Pickled classes list
intents.json: JSON file containing intents, patterns, and responses


Here's the complete README.md content as a single file:

markdown
Copy code
# 🤖 Chatbot Using NLP and Streamlit

This project is a chatbot application built using **Natural Language Processing (NLP)** and deployed with **Streamlit**. The chatbot predicts user intents and generates responses based on a pre-trained model.

## 🛠 Features

- Interactive chatbot interface
- Intent classification using a trained TensorFlow model
- Modular design for extensibility
- Sidebar navigation for additional features like conversation history and app information
- Custom styling for a better user experience

## 🧰 Technologies Used

- **Python**: Core programming language
- **TensorFlow**: Machine learning model for intent classification
- **Streamlit**: Web framework for building interactive user interfaces
- **NLTK**: Natural language processing toolkit for tokenization and lemmatization

## 🚀 How to Run the Project

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Required Python libraries (install via `requirements.txt`)

### Installation
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download NLTK data:

bash
Copy code
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
Ensure you have the following files:

chatbot_model.h5: Trained TensorFlow model
words.pkl: Pickled words list
classes.pkl: Pickled classes list
intents.json: JSON file containing intents, patterns, and responses
Run the Application
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the app in your browser at http://localhost:8501.

📂 Project Structure

├── app.py                 # Streamlit app code
├── chatbot_model.h5       # Trained TensorFlow model
├── words.pkl              # Pickled words list
├── classes.pkl            # Pickled classes list
├── intents.json           # JSON file with intents, patterns, and responses
├── requirements.txt       # Project dependencies
├── README.md              # Documentation


