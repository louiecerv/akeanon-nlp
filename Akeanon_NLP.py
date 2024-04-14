import base64
import google.generativeai as genai
import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def app():
    st.title("Akeanon NLP")
    st.write("This app uses Google's Natural Language Processing API to analyze Akeanon text.")
    model_list = []
    
    for i, m in zip(range(5), genai.list_tuned_models()):
        model_list.append(m.name)

    if len(model_list) > 0:
        selected_model = st.selectbox("Select a model", model_list)

    if st.button("Analyze Text"):
        model = genai.GenerativeModel(model_name=selected_model)
        text_input = st.text_area("Enter Akeanon text here:")
        if st.button("Translate to English"):
            result = model.generate_content(text_input)
            st.write(result.text)
        result = model.generate_content(text_input)
        st.write(result.text)

    if st.button("Create Tuned Model"):
        base_model = [
            m for m in genai.list_models()
            if "createTunedModel" in m.supported_generation_methods][0]

        st.write(base_model)
        df = pd.read_csv('./akeanon-sentences.csv', header=0, index_col=None)
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
            batch_size=10,
            learning_rate=0.001,
        )

        model = genai.get_tuned_model(f'tunedModels/{name}')
        st.write(model)
        
        for status in operation.wait_bar():
            time.sleep(1)
            st.write(operation.metadata)

        model = operation.result()
        snapshots = pd.DataFrame(model.tuning_task.snapshots)
        fig, ax = plt.subplots()
        sns.lineplot(data=snapshots, x = 'epoch', y='mean_loss')
        st.pyplot(fig)
    
    st.write("Powered by Google Cloud Natural Language API")

#run the app
if __name__ == "__main__":
  app()
