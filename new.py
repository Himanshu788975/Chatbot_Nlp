

import nltk


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')  

from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
import numpy as np
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open(r'C:\Users\Dell\Desktop\ccbot\intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordlist = nltk.word_tokenize(pattern)  # Word tokenization
        words.extend(wordlist)
        documents.append((wordlist, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    
  
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.h5')
print('Model training complete!')


