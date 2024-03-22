from transformers import pipeline as hf_pipeline
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from gtts import gTTS
from io import BytesIO

# Initialiser le pipeline de génération de texte avec microsoft/phi-2
phi_pipeline = hf_pipeline("text-generation", model="microsoft/phi-2", trust_remote_code=True)

# Fonction pour générer une image à partir d'une chaîne base64
def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

# Fonction pour générer un résumé
def generate_summary(data):
    context = "This is a dataset that contains several columns. "
    sample_data = data.sample(n=3).to_csv(index=False)  
    prompt_intro = "Given the sample data and the columns described, provide a comprehensive summary highlighting key insights, trends, and any interesting findings."
    input_text = f"{context}Here are the column names: {', '.join(data.columns)}. Sample data:\n{sample_data}\n{prompt_intro}"
    result = phi_pipeline(input_text, max_length=400)[0]['generated_text']
    return result

# Fonction pour générer un graphique
def generate_graph(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=df.columns[0])
    plt.title("Distribution")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img

# Interface Streamlit
st.title("Data Summary and Visualization Application")

menu = st.sidebar.selectbox("Choose an Option", ["Summary", "Question based Graph"])

if menu == "Summary":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Sample of the CSV Data")
        st.dataframe(df.sample(n=5))  
        summary = generate_summary(df)
        st.subheader("Generated Summary")
        st.write(summary)
        
        # Ajouter un bouton pour lire la réponse vocalement
        if st.button("Read Summary Aloud"):
            tts = gTTS(summary, lang='en')
            audio_data = BytesIO()
            tts.write_to_fp(audio_data)
            audio_data.seek(0)
            st.audio(audio_data, format='audio/mp3')

elif menu == "Question based Graph":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if st.button("Generate Graph"):
            img = generate_graph(df)
            st.image(base64_to_image(img), caption="Generated Graph")
