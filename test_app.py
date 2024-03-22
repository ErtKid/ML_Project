import pytest
from app_phi2 import base64_to_image, generate_summary, generate_graph
from PIL import Image
import pandas as pd

# Test pour la fonction base64_to_image
def test_base64_to_image():
    # Générer un exemple de base64 pour une image rouge 1x1 px
    base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAAIRAQM="
    image = base64_to_image(base64_str)
    assert isinstance(image, Image.Image), "La fonction doit retourner un objet Image."

# Test pour la fonction generate_summary
def test_generate_summary():
    # Créer un dataframe de test
    df_test = pd.DataFrame({
        'A': ['Data1', 'Data2', 'Data3'],
        'B': ['Info1', 'Info2', 'Info3']
    })
    summary = generate_summary(df_test)
    assert isinstance(summary, str), "La fonction doit retourner une chaîne de caractères."
    assert "Data1" in summary or "Data2" in summary or "Data3" in summary, "Le résumé doit contenir des éléments du dataframe."

# Test pour la fonction generate_graph (test de base)
def test_generate_graph():
    # Créer un dataframe de test
    df_test = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    graph_img_base64 = generate_graph(df_test)
    assert isinstance(graph_img_base64, str), "La fonction doit retourner une chaîne de caractères."
