import base64
import google.generativeai as genai
import streamlit as st
import os
import time
import pandas as pd
import seaborn as sns



def handle_gemini_response(response):
  try:
    # Try using the quick text accessor
    return response.text
  except ValueError as e:
    if "response.text" in str(e) and "no valid Part" in str(e):
      # Access safety ratings for debugging (optional)
      print(f"Error: No valid response part. Safety Ratings: {response.prompt_feedback.safety_ratings}")
      # Handle the error here. You can retry with a different prompt, 
      # return a default value, or raise a new exception.
      raise Exception("No valid response received from Gemini. Consider revising the prompt.")
    else:
      # Raise the original error for unexpected exceptions
      raise e
    
def app():
    st.title("Akeanon NLP")
    st.write("This app uses Google's Natural Language Processing API to analyze Akeanon text.")
    model_list = []
    
    for i, m in zip(range(5), genai.list_tuned_models()):
        model_list.append(m.name)
    if len(model_list) > 0:
        selected_model = st.selectbox("Select a model", model_list)
        model = genai.GenerativeModel(model_name=selected_model)
    if st.button("Delete Model"):
        genai.delete_tuned_model(selected_model)

    text_input = st.text_area("Enter Akeanon text here:")
    if st.button("Translate to English"):
        try:
            result = model.generate_content(f'translate to English: {text_input}')
            text = handle_gemini_response(result)
            st.write(result.text)
        except Exception as e:
            st.write(f"An error occured: {e}")

    if st.button("Translate to Akeanon"):
        try:
            result = model.generate_content(f'translate to Akeanon: {text_input}')
            text = handle_gemini_response(result)
            st.write(result.text)
        except Exception as e:
            st.write(f"An error occured: {e}")

    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
