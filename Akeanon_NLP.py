import base64
import google.generativeai as genai
import streamlit as st
import os
import time

#GOOGLE_API_KEY=st.secrets["GOOGLE_API_KEY"]
#genai.configure(api_key=GOOGLE_API_KEY)

def app():
    st.title("Akeanon NLP")
    st.write("This app uses Google's Natural Language Processing API to analyze Akeanon text.")

    for i, m in zip(range(5), genai.list_tuned_models()):
        st.write(m.name)  

    base_model = [
        m for m in genai.list_models()
        if "createTunedModel" in m.supported_generation_methods][0]

    st.write(base_model)
     
    Model(name='models/gemini-1.0-pro-001',
        base_model_id='',
        version='001',
        display_name='Gemini 1.0 Pro 001 (Tuning)',
        description=('The best model for scaling across a wide range of tasks. This is a stable '
                    'model that supports tuning.'),
        input_token_limit=30720,
        output_token_limit=2048,
        supported_generation_methods=['generateContent', 'countTokens', 'createTunedModel'],
        temperature=0.9,
        top_p=1.0,
        top_k=1)  
   
    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
