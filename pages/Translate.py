import base64
import google.generativeai as genai
import streamlit as st
import os
import time
import pandas as pd
import seaborn as sns

#GOOGLE_API_KEY=st.secrets["GOOGLE_API_KEY"]
#genai.configure(api_key=GOOGLE_API_KEY)

def app():
    st.title("Akeanon NLP")
    st.write("This app uses Google's Natural Language Processing API to analyze Akeanon text.")
    model_list = []
    
    for i, m in zip(range(5), genai.list_tuned_models()):
        model_list.append(m.name)
    selected_model = st.selectbox("Select a model", model_list)
    model = genai.GenerativeModel(model_name=selected_model)
    text_input = st.text_area("Enter Akeanon text here:")
    if st.button("Translate to English"):
        result = model.generate_content(text_input)
        st.write(result.text)
        result = model.generate_content(text_input)
        st.write(result.text)


    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
