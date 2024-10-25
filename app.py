from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import nltk
from keras.models import load_model

model_path = r'C:\Users\AUDRIK\OneDrive\Desktop\finance_dataset\Financial_sentiment_LSTM_model.h5'
MODEL = load_model(model_path)

from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

app = Flask(__name__)



# Define predict function
def predict(text, tokenizer, model):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in punctuation]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    processed_text = " ".join(lemmatized_tokens)
    
    sequences = tokenizer.texts_to_sequences([processed_text]) 
    predict_padded = pad_sequences(sequences, maxlen=500, padding='post')

    # Model prediction
    predicted_sentiment = model.predict(predict_padded)
    
    # Convert probability to binary class (0 or 1)
    predicted_class = int(np.round(predicted_sentiment[0][0]))  # Round to get either 0 or 1
    
    if (predicted_class == 0):
        return "Negative"
    else:
        return "Positive"

# Set up routes
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     text = request.form['text']
#     sentiment = predict(text, tokenizer, MODEL)
#     return jsonify({'sentiment': sentiment})

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        # Call your predict function
        sentiment = predict(text, tokenizer, MODEL)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
