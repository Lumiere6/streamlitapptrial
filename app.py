# -*- coding: utf-8 -*-
import string
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import nltk
import requests
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import tokenizer_from_json
import lime
import lime.lime_text
from lime.lime_text import LimeTextExplainer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load models and tokenizers
audio_phish_model = pickle.load(open('audiophish.sav', 'rb'))  
smishing_model = tf.keras.models.load_model('smishing_finalgood_model.h5')  
website_model = tf.keras.models.load_model('url_final_model.h5')  

# Function to clean text for different phishing detection types
def clean_text_vishing(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([w for w in text if not w.isdigit()])
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    return text

def clean_text_sms(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    text = ' '.join(tokens)
    return text

# Function to load tokenizer from URL
def load_tokenizer_from_url(url):
    response = requests.get(url)
    tokenizer_json = response.text
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

# LIME Explainer setup for Audio Phishing
explainer_audio = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])

# LIME Explainer setup for Smishing
explainer_smishing = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])

# LIME Explainer setup for Website Phishing
explainer_website = LimeTextExplainer(class_names=['Legitimate', 'Phishing'])

# Streamlit app
st.set_page_config(page_title="Phishing Detection Framework", layout="wide", page_icon="🕵️‍♀️")

with st.sidebar:
    selected = option_menu('Phishing Detection System',
                           ['Audio Phishing', 'Smishing', 'Website Phishing'],
                           icons=['loud_sound', '✉️', '🌐'],  # Icons corresponding to each category
                           default_index=0)
    
class_names = ['Legitimate', 'Phishing']

if selected == 'Audio Phishing':
    st.title('Audio Phishing Detection')
    transcript = st.text_area('Call Transcript', 'Enter call transcript here...')
    
    if st.button('Analyze Transcript'):
        cleaned_transcript = clean_text_vishing(transcript)
        
        # Tokenize and pad sequences
        with open("vishing_tokenizer.json", "r") as json_file:
            json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_transcripts = tokens.texts_to_sequences([cleaned_transcript])
        X = pad_sequences(tokenized_transcripts, maxlen=100, padding='post')
        
        # Predict using the audio phishing model
        pred = audio_phish_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.5
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
        
        # Explain the prediction using LIME
        if st.button("Explain Prediction"):
            def predict_proba(text_list):
                sequence = tokens.texts_to_sequences(text_list)
                sequence = pad_sequences(sequence, maxlen=100, padding='post')
                prediction = audio_phish_model.predict(sequence)
                return prediction
            
            exp = explainer_audio.explain_instance(cleaned_transcript, predict_proba, num_features=10)
            exp_dict = exp.as_list()
            features = [x[0] for x in exp_dict]
            weights = [x[1] for x in exp_dict]
            
            # Plot LIME explanation as a bar chart
            st.subheader('LIME Explanation:')
            st.bar_chart({features[i]: weights[i] for i in range(len(features))})

elif selected == 'Smishing':
    st.title('Smishing Detection')
    sms = st.text_area('SMS', 'Enter SMS text here...')
    
    if st.button('Analyze SMS'):
        cleaned_sms = clean_text_sms(sms)
        
        # Tokenize and pad sequences
        with open("tokenizer_smish.json", "r") as json_file:
            json_string = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_text = tokenizer.texts_to_sequences([cleaned_sms])
        X = pad_sequences(tokenized_text, maxlen=40, padding='post')
        
        # Predict using the smishing model
        pred = smishing_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.7
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
        
        # Explain the prediction using LIME
        if st.button("Explain Prediction"):
            def predict_proba(text_list):
                sequence = tokenizer.texts_to_sequences(text_list)
                sequence = pad_sequences(sequence, maxlen=40, padding='post')
                prediction = smishing_model.predict(sequence)
                return prediction
            
            exp = explainer_smishing.explain_instance(cleaned_sms, predict_proba, num_features=10)
            exp_dict = exp.as_list()
            features = [x[0] for x in exp_dict]
            weights = [x[1] for x in exp_dict]
            
            # Plot LIME explanation as a bar chart
            st.subheader('LIME Explanation:')
            st.bar_chart({features[i]: weights[i] for i in range(len(features))})

elif selected == 'Website Phishing':
    st.title('Website Phishing Detection')
    url = st.text_input('Enter URL', 'https://example.com')
    
    if st.button('Analyze URL'):
        with open("url_tokenizer.json", "r") as json_file:
            json_string = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_text = tokenizer.texts_to_sequences([url])
        X = pad_sequences(tokenized_text, maxlen=60, padding='post')
        
        # Predict using the website phishing model
        pred = website_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.7
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
        
        # Explain the prediction using LIME
        if st.button("Explain Prediction"):
            def predict_proba(text_list):
                sequence = tokenizer.texts_to_sequences(text_list)
                sequence = pad_sequences(sequence, maxlen=60, padding='post')
                prediction = website_model.predict(sequence)
                return prediction
            
            exp = explainer_website.explain_instance(url, predict_proba, num_features=10)
            exp_dict = exp.as_list()
            features = [x[0] for x in exp_dict]
            weights = [x[1] for x in exp_dict]
            
            # Plot LIME explanation as a bar chart
            st.subheader('LIME Explanation:')
            st.bar_chart({features[i]: weights[i] for i in range(len(features))})
