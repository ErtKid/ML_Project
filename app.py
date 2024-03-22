import streamlit as st
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Choix d'un modèle alternatif pour la génération de résumés
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def generate_summary(df):
    # Analyse préliminaire pour comprendre le dataset
    context_intro = "This dataset encompasses various metrics and insights. "
    description = "Key areas include "
    column_descriptions = ', '.join(df.describe().columns) + ". "  # Se concentrer sur les colonnes quantitatives pour le résumé
    overview = "This summary aims to provide a clear overview of key patterns, trends, and any anomalies present."
    prompt = context_intro + description + column_descriptions + overview
    
    # Ajustement des paramètres de génération du résumé
    summary_results = summarization_pipeline(prompt, max_length=150, min_length=40, length_penalty=2.5, no_repeat_ngram_size=3)
    summary_text = summary_results[0]['summary_text']
    return summary_text

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
        summary = generate_summary(df)
        st.write("Generated Summary:", summary)
elif menu == "Question based Graph":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if st.button("Generate Graph"):
            img = generate_graph(df)
            st.image(base64_to_image(img), caption="Generated Graph")
