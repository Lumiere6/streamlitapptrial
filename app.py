import string
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import nltk
import requests
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
from keras.preprocessing.text import tokenizer_from_json
st.set_page_config(page_title="Phishing detection Framework",layout="wide",page_icon="🕵️‍♀️")
# model

audio_phish_model = pickle.load(open('audiophish.sav', 'rb'))
smishing_model = tf.keras.models.load_model('smishing_finalgood_model.h5')
website_model = tf.keras.models.load_model('url_final_model.h5')
email_model=tf.keras.models.load_model('email_model.h5')
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
def load_tokenizer_from_url(url):
    response = requests.get(url)
    tokenizer_json = response.text
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer
  
with st.sidebar:
    selected = option_menu('Phishing detection system',
                          ['Audio Phishing',
                           'Email Phishing',
                           'Website phishing',
                           'Smishing'],
                          icons=['loud_sound','🔉','','✉️'],
                          default_index=0)
class_names=[' Phishing','Legitimate']

if selected == 'Audio Phishing':
    st.title('Audio Phishing Detection')
    transcript = st.text_input('Call Transcript')
    if st.button("Analyze Transcript"):
        cleaned_transcript = clean_text_vishing(transcript)
        with open("vishing_tokenizer.json", "r") as json_file:
            json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_transcripts = tokens.texts_to_sequences([cleaned_transcript])
        X = pad_sequences(tokenized_transcripts, maxlen=100, padding='post')
        pred = audio_phish_model.predict(X)
        
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.5
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
    if st.button("Explain Prediction"):
      def predict_proba(text):
        with open("vishing_tokenizer.json", "r") as json_file:
          json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        sequence = tokens.texts_to_sequences(text)
        sequence = pad_sequences(sequence, maxlen=100, padding='post')
        prediction = audio_phish_model.predict(sequence)
        returnable = []
        for i in prediction:
          temp = i[0]
          returnable.append(np.array([1-temp, temp]))
        return np.array(returnable)
      explainer= LimeTextExplainer(class_names=class_names)
      exp = explainer.explain_instance(clean_text_vishing(transcript),predict_proba)
      st.subheader('LIME Explanation:')
      exp_dict = exp.as_list()
      features = [x[0] for x in exp_dict]
      weights = [x[1] for x in exp_dict]
# Plot bar chart using Streamlit
      st.bar_chart({features[i]: weights[i] for i in range(len(features))})
    
        

if selected == 'Smishing':
    st.title("Smishing Detection")
    sms = st.text_input("SMS")
    if st.button("Analyze SMS"):
        cleaned_sms = clean_text_sms(sms)
        with open("tokenizer_smish.json", "r") as json_file:
            json_string = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_text = tokenizer.texts_to_sequences([cleaned_sms])
        X = pad_sequences(tokenized_text,maxlen=40,padding='post')
        
        pred = smishing_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.7
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
      
    if st.button("Explain Prediction"):
      def predict_proba(text):
        with open("tokenizer_smish.json", "r") as json_file:
          json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        sequence = tokens.texts_to_sequences(text)
        sequence = pad_sequences(sequence, maxlen=40, padding='post')
        prediction = smishing_model.predict(sequence)
        returnable = []
        for i in prediction:
          temp = i[0]
          returnable.append(np.array([1-temp, temp]))
        return np.array(returnable)
      explainer= LimeTextExplainer(class_names=class_names)
      exp = explainer.explain_instance(clean_text_sms(sms),predict_proba)
      st.subheader('LIME Explanation:')
      exp_dict = exp.as_list()
      features = [x[0] for x in exp_dict]
      weights = [x[1] for x in exp_dict]
# Plot bar chart using Streamlit
      st.bar_chart({features[i]: weights[i] for i in range(len(features))})
    
      
if selected == 'Website phishing':
    st.title("Website Phishing Detection")
    url = st.text_input("Enter URL")
    if st.button("Analyze URL"):
        with open("url_tokenizer.json", "r") as json_file:
            json_string = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_text = tokenizer.texts_to_sequences([url])
        X = pad_sequences(tokenized_text, maxlen=60, padding='post')
        
        pred = website_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.7
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
    if st.button("Explain Prediction"):
      def predict_proba(text):
        with open("url_tokenizer.json", "r") as json_file:
          json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        sequence = tokens.texts_to_sequences(text)
        sequence = pad_sequences(sequence, maxlen=60, padding='post')
        prediction = audio_phish_model.predict(sequence)
        returnable = []
        for i in prediction:
          temp = i[0]
          returnable.append(np.array([1-temp, temp]))
        return np.array(returnable)
      explainer= LimeTextExplainer(class_names=class_names)
      exp = explainer.explain_instance(url,predict_proba)
      st.subheader('LIME Explanation:')
      exp_dict = exp.as_list()
      features = [x[0] for x in exp_dict]
      weights = [x[1] for x in exp_dict]
# Plot bar chart using Streamlit
      st.bar_chart({features[i]: weights[i] for i in range(len(features))})
    
  
if selected == 'Email Phishing':
    st.title("Email Phishing Detection")
    email = st.text_input("Email content")
    if st.button("Analyze Email"):
        with open("email_tokenizer.json", "r") as json_file:
            json_string = json_file.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        tokenized_text = tokenizer.texts_to_sequences([email])
        X = pad_sequences(tokenized_text, maxlen=1323, padding='post')
        pred = email_model.predict(X)
        max_pred = np.max(pred)
        if max_pred >= 0.5:
            average_prediction = 0.7
        else:
            average_prediction = np.mean(pred, axis=0)
        prediction = "The text is predicted to be: " + class_names[np.argmax(average_prediction)]
        st.success(prediction)
      
    if st.button("Explain Prediction"):
      def predict_proba(text):
        with open("email_tokenizer.json", "r") as json_file:
          json_string = json_file.read()
        tokens = tf.keras.preprocessing.text.tokenizer_from_json(json_string)
        sequence = tokens.texts_to_sequences(text)
        sequence = pad_sequences(sequence, maxlen=1323, padding='post')
        prediction = email_model.predict(sequence)
        returnable = []
        for i in prediction:
          temp = i[0]
          returnable.append(np.array([1-temp, temp]))
        return np.array(returnable)
      explainer= LimeTextExplainer(class_names=class_names)
      exp = explainer.explain_instance(clean_text_email(email),predict_proba)
      st.subheader('LIME Explanation:')
      exp_dict = exp.as_list()
      features = [x[0] for x in exp_dict]
      weights = [x[1] for x in exp_dict]
# Plot bar chart using Streamlit
      st.bar_chart({features[i]: weights[i] for i in range(len(features))})
    
