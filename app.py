# -*- coding: utf-8 -*-

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import nltk
from keras.models import load_model
                       
audio_phish_model =load_model('audio_phish_model.h5')

#spoof_model = pickle.load(open('audiospoof.sav', 'rb'))

#email_model = pickle.load(open('email.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Phishing detection system',
                          ['Audio Phishing',
                           'Audio Spoofing',
                           'Email Phishing'],
                          icons=['music','music','mail'],
                          default_index=0)


# Diabetes Prediction Page
if (selected == 'Audio Phishing'):
    # page title
    st.title('Audio Phishing detection')
    # getting the input data from the user
    col1 = st.columns(1)
    with col1:
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


if st.button('Diabetes Test Result'):
    prediction = audio_phish_model.predict(Transcripts)

    if (prediction[0] >= 0.5):
        phish_detection = 'phish'
    else:
        phish_detection = 'legitimate'

    st.success(audio_phish_detection)
