# Utiliser une image Python officielle comme base
FROM python:3.9

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de dépendances et installer les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copier le reste de votre code d'application dans le conteneur
COPY . .

# Exécuter l'application
CMD ["streamlit", "run", "app_phi2.py", "--server.port=8501"]
