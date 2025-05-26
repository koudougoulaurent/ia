import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import load_data, preprocess_data, train_models, evaluate_models

st.set_page_config(page_title="Analyse des Maladies Cardiaques", layout="wide")

st.title("Analyse et Prédiction des Maladies Cardiaques")
st.write("Ce projet utilise le dataset Heart Disease UCI pour prédire la présence de maladies cardiaques.")

# Chargement et prétraitement des données
@st.cache_data
def load_and_preprocess():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    return df, X_train, X_test, y_train, y_test

df, X_train, X_test, y_train, y_test = load_and_preprocess()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page", ["Exploration des Données", "Modèles de Machine Learning", "Prédiction"])

if page == "Exploration des Données":
    st.header("Exploration des Données")
    
    # Statistiques descriptives
    st.subheader("Statistiques Descriptives")
    st.write(df.describe())
    
    # Distribution de l'âge
    st.subheader("Distribution de l'Âge")
    fig_age = px.histogram(df, x='age', color='target',
                          title='Distribution de l\'âge par présence de maladie cardiaque',
                          labels={'age': 'Âge', 'target': 'Maladie Cardiaque'})
    st.plotly_chart(fig_age)
    
    # Analyse par sexe
    st.subheader("Analyse par Sexe")
    sex_stats = df.groupby(['sex', 'target']).size().unstack()
    fig_sex = px.bar(sex_stats, title='Distribution des maladies cardiaques par sexe',
                    labels={'value': 'Nombre de patients', 'sex': 'Sexe (1=Homme, 0=Femme)'})
    st.plotly_chart(fig_sex)
    
    # Analyse des douleurs thoraciques
    st.subheader("Analyse des Douleurs Thoraciques")
    cp_stats = df.groupby(['cp', 'target']).size().unstack()
    fig_cp = px.bar(cp_stats, title='Relation entre type de douleur thoracique et maladie cardiaque',
                   labels={'value': 'Nombre de patients', 'cp': 'Type de douleur thoracique'})
    st.plotly_chart(fig_cp)
    
    # Statistiques des variables numériques
    st.subheader("Statistiques par Groupe")
    numeric_stats = df.groupby('target').agg({
        'trestbps': ['mean', 'std'],
        'chol': ['mean', 'std'],
        'thalach': ['mean', 'std']
    }).round(2)
    st.write(numeric_stats)
    
    # Corrélations
    st.subheader("Matrice de Corrélation")
    corr = df.corr()
    fig_corr = px.imshow(corr, title='Matrice de Corrélation')
    st.plotly_chart(fig_corr)

elif page == "Modèles de Machine Learning":
    st.header("Modèles de Machine Learning")
    
    if st.button("Entraîner les Modèles"):
        with st.spinner("Entraînement des modèles en cours..."):
            # Entraînement des modèles
            models = train_models(X_train, y_train)
            
            # Évaluation des modèles
            results = evaluate_models(models, X_test, y_test)
            
            # Affichage des résultats
            st.subheader("Résultats des Modèles")
            st.dataframe(results.style.format("{:.3f}"))
            
            # Graphique des performances
            fig = go.Figure()
            for metric in results.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results.index,
                    y=results[metric]
                ))
            fig.update_layout(
                title="Comparaison des Performances des Modèles",
                xaxis_title="Modèles",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig)

else:  # Page de prédiction
    st.header("Prédiction de Maladie Cardiaque")
    
    # Formulaire pour les entrées
    st.subheader("Entrez les informations du patient")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=0, max_value=100, value=50)
        sex = st.selectbox("Sexe", ["Homme", "Femme"])
        cp = st.selectbox("Type de douleur thoracique", 
                         ["Typique", "Atypique", "Non-angineux", "Asymptomatique"])
        trestbps = st.number_input("Pression artérielle au repos (mm Hg)", 
                                 min_value=0, max_value=300, value=120)
        chol = st.number_input("Cholestérol (mg/dl)", 
                             min_value=0, max_value=600, value=200)
        fbs = st.checkbox("Glycémie à jeun > 120 mg/dl")
    
    with col2:
        restecg = st.selectbox("Résultats électrocardiographiques", 
                              ["Normal", "Anomalie ST-T", "Hypertrophie ventriculaire"])
        thalach = st.number_input("Fréquence cardiaque maximale", 
                                min_value=0, max_value=300, value=150)
        exang = st.checkbox("Angine induite par l'exercice")
        oldpeak = st.number_input("Dépression ST induite par l'exercice", 
                                min_value=0.0, max_value=10.0, value=0.0)
        slope = st.selectbox("Pente du segment ST", 
                           ["Montante", "Plate", "Descendante"])
        ca = st.number_input("Nombre de vaisseaux colorés", 
                           min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassémie", 
                          ["Normal", "Défaut fixe", "Défaut réversible"])
    
    if st.button("Prédire"):
        # Conversion des entrées
        sex = 1 if sex == "Homme" else 0
        cp_map = {"Typique": 0, "Atypique": 1, "Non-angineux": 2, "Asymptomatique": 3}
        cp = cp_map[cp]
        fbs = 1 if fbs else 0
        restecg_map = {"Normal": 0, "Anomalie ST-T": 1, "Hypertrophie ventriculaire": 2}
        restecg = restecg_map[restecg]
        exang = 1 if exang else 0
        slope_map = {"Montante": 0, "Plate": 1, "Descendante": 2}
        slope = slope_map[slope]
        thal_map = {"Normal": 3, "Défaut fixe": 6, "Défaut réversible": 7}
        thal = thal_map[thal]
        
        # Création du vecteur d'entrée
        input_data = pd.DataFrame({
            'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
            'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
            'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca],
            'thal': [thal]
        })
        
        # Prédiction avec tous les modèles
        models = train_models(X_train, y_train)
        predictions = {}
        
        for name, model in models.items():
            pred = model.predict_proba(input_data)[0][1]
            predictions[name] = pred
        
        # Affichage des résultats
        st.subheader("Résultats de la Prédiction")
        for model, prob in predictions.items():
            st.write(f"{model}: {prob:.2%} de probabilité de maladie cardiaque") 