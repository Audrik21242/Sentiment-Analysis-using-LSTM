# Sentiment Analysis using LSTM
This project aims to build an LSTM-based model to predict the sentiment of financial news articles. The model classifies news headlines as "good" or "bad," helping to gauge the general sentiment in the financial sector.

# Table of Contents
1. Project Overview
2. Libraries and Dependencies
3. Data Preprocessing
4. Model Architecture
5. Results and Observations
6. Deployment
7. Future Improvements

# Project Overview
The dataset used consists of financial news headlines labeled as either "good" or "bad" news. Initially, data from two separate datasets was combined to form a larger dataset. However, it was found to be imbalanced, with 4,526 good news headlines and 1,463 bad news headlines. The imbalance was addressed in the preprocessing stage to ensure balanced learning.

# Libraries and Dependencies
The project uses the following libraries:

 * NumPy and Pandas: For data manipulation and preprocessing.
 * NLTK and string: For text processing (tokenization, lemmatization, stopword removal).
 * Imbalanced-Learn (SMOTE): To balance the dataset.
 * TensorFlow: For building and training the LSTM model.
 * Flask: For creating a simple web-based frontend.
 * JSON and OS: For saving/loading the tokenizer and model.

# Data Preprocessing
Text Processing:
 * Tokenization: Converting text into sequences of words.
 * Lemmatization: Reducing words to their base forms.
 * Stopword & Punctuation Removal: Cleaning text to retain only essential information.
 * Vectorization: Text was converted into numerical vectors, with each entry representing a unique word. Padding was applied to standardize vector lengths across all samples.
 * Handling Imbalance: SMOTE was applied separately to training, validation, and test sets to avoid data leakage.
 * Data Splitting: The data was divided into training, validation, and test sets.

# Model Architecture
The model is a stacked LSTM network designed to handle sequence-based text data:

 * Embedding Layer: Handles word embeddings for input processing.
 * Three LSTM Layers: Capture sequential patterns and sentiment dependencies.
 * Dense Layers: For sentiment classification.
The model was trained for 25 epochs, as further training led to overfitting.

# Results and Observations
Validation Accuracy: 0.717<br/>
Test Accuracy: 0.719<br/>

These results indicate a reasonably accurate model with similar performance on both validation and test sets, demonstrating that it generalizes well without significant overfitting.

# Deployment
A simple frontend was created using Flask, which allows users to input a news headline and get a sentiment prediction. The prediction function takes in text data, the trained model, and the tokenizer, returning either "Positive" or "Negative" based on the sentiment.

Usage
 * git clone https://github.com/Audrik21242/Sentiment-Analysis-using-LSTM.git

Run the Flask App:
 * python app.py
 * Access the app locally at http://127.0.0.1:5000.

# Future Improvements
 * Hyperparameter Tuning: Experiment with different model architectures and hyperparameters to improve performance.
 * Additional Features: Incorporate other NLP techniques like attention mechanisms for potentially better sentiment capture.
