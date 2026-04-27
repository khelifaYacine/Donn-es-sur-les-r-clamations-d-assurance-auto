🧾 README — Insurance Claims Dataset
Détection de fraude dans les sinistres d’assurance
🎯 Objectif du projet
Construire un modèle de machine learning capable de prédire si un sinistre est frauduleux (fraud_reported) à partir des informations du client, du contrat, du véhicule et des circonstances de l’incident.

Ce projet illustre une démarche complète : EDA, nettoyage, feature engineering, modélisation, évaluation et interprétation.

📂 Dataset
Le dataset contient 1000 sinistres, chacun décrit par 40 variables couvrant :

1. Informations client
months_as_customer

age

insured_sex

insured_education_level

insured_occupation

insured_hobbies

insured_relationship

capital-gains, capital-loss

2. Informations contrat
policy_number

policy_bind_date

policy_state

policy_csl

policy_deductable

policy_annual_premium

umbrella_limit

3. Informations incident
incident_date

incident_type

collision_type

incident_severity

authorities_contacted (seule colonne avec valeurs manquantes)

incident_state, incident_city, incident_location

incident_hour_of_the_day

number_of_vehicles_involved

property_damage

bodily_injuries

witnesses

police_report_available

4. Montants des sinistres
total_claim_amount

injury_claim

property_claim

vehicle_claim

5. Véhicule
auto_make

auto_model

auto_year

6. Variable cible
fraud_reported (Yes/No)

7. Colonne inutile
_c39 (100% NaN → à supprimer)

Source confirmée : dataset insurance_claims.csv publié sur Mendeley Data .

🧪 Méthodologie
1. Exploration des données (EDA)
Vérification des distributions

Analyse des corrélations entre montants et fraude

Analyse des patterns suspects :

incohérences entre incident_type et collision_type

sinistres élevés sans témoins

absence de rapport de police

2. Nettoyage
Suppression de _c39

Traitement de authorities_contacted (valeurs manquantes)

Conversion des dates

Encodage des variables catégorielles (beaucoup de colonnes → One‑Hot ou Target Encoding)

3. Feature Engineering
Dérivation de variables temporelles (jour, mois, saison)

Détection d’incohérences (ex : number_of_vehicles_involved = 1 + collision_type = Rear Collision)

Agrégation des montants (injury_claim + property_claim + vehicle_claim)

4. Modélisation
Modèles adaptés à la classification :

Logistic Regression

Random Forest

Gradient Boosting / XGBoost

Balanced models (SMOTE, class weights)

5. Évaluation
Accuracy

Precision / Recall

F1-score

Matrice de confusion

AUC-ROC

🧠 Insights attendus
Les fraudes sont souvent associées à :

absence de témoins

absence de rapport de police

montants de sinistres incohérents

collisions atypiques

Certaines professions ou hobbies peuvent apparaître corrélés (attention au biais → à documenter)

🧩 Exemple de code (extrait)
python
import pandas as pd

df = pd.read_csv("insurance_claims.csv")

# Suppression colonne vide
df = df.drop(columns=["_c39"])

# Variable cible
y = df["fraud_reported"].map({"Y":1, "N":0})
X = df.drop("fraud_reported", axis=1)
📌 Conclusion
Ce dataset est parfait pour un projet de détection de fraude, avec un large éventail de variables riches et réalistes.
Ton README doit refléter cette complexité — ce n’est pas un dataset de régression simple, mais un vrai cas métier d’assurance.

🔧 Étapes suivantes possibles
Ajouter une section Data Cleaning détaillée

Ajouter un schéma clair des catégories de variables

Construire un pipeline ML complet

Ajouter des visualisations (fraude vs non‑fraude)