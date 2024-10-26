from flask import Flask, request,render_template
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
import os
import json
from keras.preprocessing.text import tokenizer_from_json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = r'C:\Users\AUDRIK\OneDrive\Desktop\finance_dataset\model1.h5'
MODEL = load_model(model_path, custom_objects={"Bidirectional": Bidirectional})

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(json.dumps(data))

app = Flask(__name__)


def predict(text, tokenizer, model):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word not in punctuation]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    processed_text = " ".join(lemmatized_tokens)
    
    sequences = tokenizer.texts_to_sequences([processed_text]) 
    predict_padded = pad_sequences(sequences, maxlen=500, padding='post')

    with tf.device('/CPU:0'):
        predicted_sentiment = model.predict(predict_padded)
    predicted_class = int(np.round(predicted_sentiment[0][0]))
    
    return "Positive" if predicted_class == 1 else "Negative"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict(text, tokenizer, MODEL)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run()
