import base64
import google.generativeai as genai
import streamlit as st
import os
import time
import pandas as pd


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
    df = pd.read_csv('./akeanon-words.csv', header=0)
    df.reset_index(drop=True)
    st.write(df)

    # Convert to dictionary list
    dict_list = df.to_dict('records')

    st.write(dict_list)
    
    import random
    name = f'generate-num-{random.randint(0,10000)}'

    genai.create_tuned_model(
        source_model=base_model.name,
        training_data=[dict_list],
        id = name,
        epoch_count = 100,
        batch_size=4,
        learning_rate=0.001,
    )

    model = genai.get_tuned_model(f'tunedModels/{name}')
    st.write(model)
     
    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
