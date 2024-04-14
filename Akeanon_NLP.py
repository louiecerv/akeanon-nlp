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

    for i, m in zip(range(5), genai.list_tuned_models()):
        st.write(m.name)  

    base_model = [
        m for m in genai.list_models()
        if "createTunedModel" in m.supported_generation_methods][0]

    st.write(base_model)
    df = pd.read_csv('./akeanon-words.csv', header=0, index_col=None)
    df = df.reset_index(drop=True)
    st.write(df)

    # Convert DataFrame to a list of dictionaries
    data_list = df.to_dict(orient='records')

    import random
    name = f'generate-num-{random.randint(0,10000)}'

    operation = genai.create_tuned_model(
        source_model=base_model.name,
        training_data=data_list,
        id = name,
        epoch_count = 10,
        batch_size=64,
        learning_rate=0.01,
    )

    model = genai.get_tuned_model(f'tunedModels/{name}')
    st.write(model)
     
    for status in operation.wait_bar():
        time.sleep(30)

    model = operation.result()
    snapshots = pd.DataFrame(model.tuning_task.snapshots)
    sns.lineplot(data=snapshots, x = 'epoch', y='mean_loss')

        
    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
