import streamlit as st
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Initialisation du pipeline GPT-2 pour la génération de texte
gpt2_pipeline = pipeline("text-generation", model="gpt2")

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def generate_summary(data):
    # Préparer une introduction plus générale
    prompt = ("This dataset includes the following features: " +
              ", ".join(data.columns) + ". " +
              "It covers various countries and their happiness scores, " +
              "including aspects like GDP per capita, social support, " +
              "and perceptions of corruption. " +
              "Can you provide an insightful summary based on these features?")
    
    # Génération du résumé avec GPT-2
    summary = gpt2_pipeline(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9, num_return_sequences=1)[0]['generated_text']
    # La fonction clean_summary peut être utilisée ici si nécessaire
    return summary



def clean_summary(summary, prompt):
    # Supprime la requête du début du résumé généré
    if summary.startswith(prompt):
        return summary[len(prompt):].lstrip()
    # Génération du résumé avec GPT-2
    summary = gpt2_pipeline(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, num_return_sequences=1)[0]['generated_text']
    # Nettoyage du résumé en enlevant la requête initiale
    summary = clean_summary(summary, prompt)
    return summary

def generate_graph(df):
    # Exemple simple de génération de graphique avec seaborn/matplotlib
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=df.columns[0])  # Exemple basé sur la première colonne
    plt.title("Distribution")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img

# Interface Streamlit
st.title("Application de Résumé et de Visualisation avec GPT-2")

menu = st.sidebar.selectbox("Choisir une option", ["Résumé", "Graphique basé sur une question"])

if menu == "Résumé":
    uploaded_file = st.file_uploader("Télécharger votre fichier CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        summary = generate_summary(df)
        st.write("Résumé généré :", summary)
elif menu == "Graphique basé sur une question":
    uploaded_file = st.file_uploader("Télécharger votre fichier CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if st.button("Générer le Graphique"):
            img = generate_graph(df)
            st.image(base64_to_image(img), caption="Graphique généré")
