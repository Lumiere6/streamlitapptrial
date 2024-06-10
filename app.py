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
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import lime
import lime.lime_text
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Phishing detection Framework",
                   layout="wide",
                   page_icon="🕵️‍♀️")
# model
audio_phish_model = pickle.load(open('audiophish.sav', 'rb'))

#preprocessing functions

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

def clean_text_email(text):
  text=str(text)
  text = text.lower()
  tokens = word_tokenize(text)
  tokens = [token for token in tokens if token not in string.punctuation]
  text = ' '.join(tokens)
  return text

def clean_text_sms(text):
  text=str(text)
  text = text.lower()
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if token not in stop_words]
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tokens]
  text = ' '.join(tokens)
  return text

with st.sidebar:
    selected = option_menu('Phishing detection system',
                          ['Audio Phishing',
                           'Audio Spoofing',
                           'Email Phishing',
                           'Website phishing',
                           'Smishing'],
                          icons=['loud_sound','🔉','📧','','✉️'],
                          default_index=0)

if (selected == 'Audio Phishing'):
    st.title('Audio Phishing detection')
    Transcripts = st.text_input('Call Transcript')
    cleaned_transcripts=clean_text_vishing(Transcripts)

    with open("vishing_tokenizer.json", "r") as json_file:
      json_string = json_file.read()
    tokens=tf.keras.preprocessing.text.tokenizer_from_json(json_string)
    tokenized_transcripts=tokens.texts_to_sequences(cleaned_transcripts)
    X = pad_sequences(tokenized_transcripts,maxlen=100,padding='post')
    pred=audio_phish_model.predict(X)

  
class_names=[0,1]
explainer= LimeTextExplainer(class_names=class_names)

def predict_proba(text):
  sequence=tokens.texts_to_sequences(text)
  sequence=pad_sequences(sequence,maxlen=100,padding='post')
  prediction=audio_phish_model.predict(X)
  returnable=[]
  for i in prediction:
    temp=i[0]
    returnable.append(np.array([1-temp,temp]))
  return np.array(returnable)

explainer.explain_instance(Transcripts,predict_proba)

if (selected == 'Email Phishing'):
    st.title('Email Phishing')
    email = st.text_input('Email')

if (selected == 'Website phishing'):
    st.title('Website Phishing')
    website = st.text_input('URL')


if (selected == 'Smishing'):
    st.title('Smishing')
    sms = st.text_input('SMS')

if st.button('Result'):
  prediction = audio_phish_model.predict(X)

st.echo(prediction)
