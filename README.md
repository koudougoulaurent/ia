# Projet d'Analyse et Prédiction des Maladies Cardiaques

Ce projet utilise des techniques de machine learning pour analyser et prédire la présence de maladies cardiaques à partir du dataset Heart Disease UCI.

## Description du Projet

Le projet comprend :
- Une analyse exploratoire des données
- L'implémentation de plusieurs algorithmes de classification
- Une interface utilisateur interactive avec Streamlit
- Une comparaison des performances des différents modèles

## Installation

1. Clonez ce dépôt
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application :
```bash
streamlit run app.py
```

## Structure du Projet

- `app.py` : Application Streamlit pour l'interface utilisateur
- `model.py` : Code de machine learning et fonctions de prétraitement
- `requirements.txt` : Dépendances du projet

## Fonctionnalités

1. **Exploration des Données**
   - Statistiques descriptives
   - Visualisations interactives
   - Analyse des corrélations

2. **Modèles de Machine Learning**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - AdaBoost

3. **Métriques d'Évaluation**
   - Accuracy
   - Précision
   - Rappel
   - F1-Score
   - AUC-ROC

## Dataset

Le dataset utilisé est le Heart Disease UCI Dataset, qui contient des informations sur des patients et indique si une personne est atteinte d'une maladie cardiaque. 