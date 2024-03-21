import streamlit as st
from transformers import pipeline
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Initialiser le pipeline de génération de texte avec microsoft/phi-2
phi_pipeline = pipeline("text-generation", model="microsoft/phi-2", trust_remote_code=True)

def base64_to_image(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

def generate_summary(data):
    # Exemple de contexte et de données pour le prompt
    context = "This is a dataset that contains severals collums. "
    sample_data = data.sample(n=3).to_csv(index=False)  # Prenez un échantillon de données
    prompt_intro = "Given the sample data and the columns described, provide a comprehensive summary highlighting key insights, trends, and any interesting findings."
    
    # Construction du prompt
    input_text = f"{context}Here are the column names: {', '.join(data.columns)}. Sample data:\n{sample_data}\n{prompt_intro}"
    
    # Génération du résumé avec le modèle
    result = phi_pipeline(input_text, max_length=400)[0]['generated_text']
    return result


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
st.title("Data Summary and Visualization Application with Phi-2")

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
