import streamlit as st
import pickle
import numpy as np

def load_model():
  with open('audiophish.pkl',rb) as file:
    data = pickle.load(file
  return data

data=load_model()

def predict_page():
  st.title("Vishing prediction")
