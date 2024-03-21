import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Chargez le tokenizer et le modèle
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Initialisez le pipeline de classification de texte avec DistilBERT
distilbert_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def generate_summary(data):
    # Adaptation pour la classification de texte (remarque : DistilBERT n'est pas un modèle de génération de texte)
    # Utilisation simplifiée pour la démo - vous devrez adapter cette partie en fonction de vos besoins réels
    summary = "Résumé basé sur l'analyse des données : \n"
    for column in data.columns:
        predictions = distilbert_pipeline("This column name " + column + " seems interesting.")
        # Prenez la prédiction la plus élevée (positif ou négatif) pour simplifier
        sentiment = max(predictions[0], key=lambda x: x['score'])
        summary += f"La colonne {column} a une sentiment {sentiment['label']}.\n"
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
st.title("Application de Résumé et de Visualisation avec DistilBERT")

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
