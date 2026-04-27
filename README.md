# Ensembles-de-donn-es-personnels-sur-le-co-t-m-dical
Prévision de l’assurance en utilisant la régression linéaire
Prédiction des coûts d’assurance médicale par régression linéaire
🎯 Objectif du projet
Construire un modèle de régression linéaire capable de prédire le coût d’assurance médicale (charges) à partir de caractéristiques individuelles (âge, BMI, tabagisme, etc.).
Ce projet illustre une démarche complète : EDA, préparation des données, modélisation, évaluation et interprétation.

📂 Dataset
Le projet utilise le Medical Cost Personal Dataset (1 338 individus, 7 variables).
Variables disponibles :

age — âge de l’assuré

sex — homme/femme

bmi — indice de masse corporelle

children — nombre d’enfants à charge

smoker — fumeur ou non

region — région de résidence

charges — coût médical facturé (variable cible)

🧪 Méthodologie
1. Exploration des données (EDA)
Analyse de la distribution des variables

Détection des outliers (notamment chez les fumeurs)

Corrélations : âge, BMI et tabagisme fortement liés aux charges

2. Préparation des données
Encodage des variables catégorielles (One‑Hot Encoding)

Normalisation non nécessaire pour la régression linéaire simple

Split train/test (80/20)

3. Modélisation
Modèles testés :

Régression linéaire multiple

Modèles régularisés (Ridge, Lasso)

Ajout de features d’interaction (ex : smoker * bmi)

4. Évaluation
Métriques utilisées :

MAE

MSE

R²

Résultats typiques :

R² autour de 0.75–0.80

MAE autour de 2100–2300

Forte amélioration avec les interactions liées au tabagisme

🧠 Insights clés
Le tabagisme est le facteur le plus déterminant dans l’augmentation des coûts.

Le BMI influence fortement les charges, surtout chez les fumeurs.

L’âge a un effet linéaire clair.

Les modèles régularisés n’apportent que peu d’amélioration.

🧩 Exemple de code (extrait)
python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

categorical = ["sex", "smoker", "region"]
numeric = ["age", "bmi", "children"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("prep", preprocess),
    ("lr", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

print("R² :", model.score(X_test, y_test))
📌 Conclusion
Ce projet montre comment un modèle simple, bien préparé et bien interprété peut fournir des prédictions fiables sur un sujet réel : les coûts d’assurance médicale.
Il constitue une base solide pour :

tester d’autres modèles (Random Forest, XGBoost)

ajouter des features non linéaires

construire un projet portfolio clair et professionnel