import streamlit as st
import requests
import pickle


def load_active_users(file_path="top_users.pkl"):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier pickle : {e}")
        return []


def get_recommendations(user_id):
    url = f"https://recommanderpremium.azurewebsites.net/api/users/{user_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return f"Erreur : {response.status_code} - {response.text}"


st.title("Système de Recommandation d'Articles")

# Obtenir la liste des ID des utilisateurs actifs
active_user_ids = load_active_users()

# Créer le menu déroulant avec les ID des utilisateurs
selected_user_id = st.selectbox(
    "Sélectionnez l'ID de l'utilisateur",
    options=active_user_ids,
    format_func=lambda x: f"Utilisateur {x}",
)

if st.button("Obtenir des recommandations"):
    recommendations = get_recommendations(selected_user_id)
    st.write(f"Recommandations pour l'utilisateur {selected_user_id}:")
    st.write(recommendations)
