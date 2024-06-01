# -*- coding: utf-8 -*-
import string
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
                       
audio_phish_model =load_model('audio_phish_model.h5')
with st.sidebar:
    selected = option_menu('Phishing detection system',
                          ['Audio Phishing',
                           'Audio Spoofing',
                           'Email Phishing'],
                          icons=['music', 'music','mail'],
                          default_index=0)


# Diabetes Prediction Page
if (selected == 'Audio Phishing'):
    # page title
    st.title('Audio Phishing detection')
    Transcripts = st.text_input('Call Transcript')

    # code for Prediction
def clean_text(text):
  
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
Transcripts=clean_text(str(Transcripts))


import json
from tensorflow.keras.preprocessing.text import Tokenizer

with open('vishing_tokenizer.json', 'r') as f:
    data = json.load(f)

tokenizer = Tokenizer()
tokenizer.from_json(data)

from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = tokenizer.texts_to_sequences(Transcripts)
X = pad_sequences(sequences,padding='post',maxlen=100)


if st.button('Result'):
    prediction = audio_phish_model.predict(X)

    if (prediction[0] >= 0.5):
        phish_detection = 'phish'
    else:
        phish_detection = 'legitimate'
