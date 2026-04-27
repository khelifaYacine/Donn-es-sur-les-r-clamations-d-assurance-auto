🧾 README — Auto Insurance Claims Data
Prédiction du montant des sinistres automobiles
🎯 Objectif du projet
Développer un modèle de régression (linéaire ou régularisée) pour prédire le montant des sinistres (ClaimAmount) à partir de caractéristiques liées au conducteur, au véhicule et à l’historique de sinistres.

Ce projet illustre une démarche complète : EDA, préparation des données, modélisation, évaluation et interprétation.

📂 Dataset
Le dataset contient des informations sur les assurés, leur véhicule et leurs sinistres.
Les colonnes visibles dans la source incluent notamment :

ClaimAmount — montant du sinistre

Age — âge du conducteur

VehicleAge — âge du véhicule

PreviousClaims — nombre de sinistres antérieurs

Gender_Male — indicateur homme

Location_Suburban / Urban — type de zone

VehicleType_SUV / Sedan / Truck — type de véhicule

Ces variables sont confirmées dans la fiche Kaggle .

🧪 Méthodologie
1. Exploration des données (EDA)
Analyse des distributions (ClaimAmount, Age, VehicleAge…)

Détection d’outliers (montants extrêmes)

Analyse des corrélations :

ClaimAmount ↗ avec PreviousClaims

ClaimAmount ↗ avec VehicleAge (tendance observée dans les exemples Kaggle)

Effets possibles du type de véhicule

2. Préparation des données
Encodage des variables catégorielles (déjà one‑hot dans le dataset)

Vérification des doublons

Split train/test (80/20)

Optionnel : transformation log du ClaimAmount si distribution très asymétrique

3. Modélisation
Modèles recommandés :

Régression linéaire multiple

Ridge / Lasso (souvent utiles si multicolinéarité entre types de véhicules)

Random Forest Regressor pour comparaison

4. Évaluation
Métriques utilisées :

MAE

RMSE

R²

🧠 Insights attendus
Les conducteurs ayant plus de sinistres antérieurs ont des montants plus élevés.

Les véhicules plus anciens peuvent générer des coûts plus importants (corrélation observée dans les données Kaggle).

Le type de véhicule (SUV, Sedan, Truck) influence le montant du sinistre.

L’âge du conducteur peut jouer un rôle non linéaire (jeunes conducteurs = risque plus élevé).

🧩 Exemple de code (extrait)
python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("cleaned_insurance_claims.csv")

X = df.drop("ClaimAmount", axis=1)
y = df["ClaimAmount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("R² :", r2_score(y_test, pred))
📌 Conclusion
Ce projet permet de :

comprendre les facteurs influençant les montants de sinistres

tester plusieurs modèles de régression

produire un projet portfolio clair et professionnel

🔧 Étapes suivantes possibles
Ajouter des interactions (ex : VehicleAge × VehicleType)

Tester des modèles non linéaires

Construire une API de prédiction

Ajouter des visualisations (heatmap, boxplots, partial dependence)

