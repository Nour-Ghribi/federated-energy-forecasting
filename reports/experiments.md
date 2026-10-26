# Registre expérimental

L'expérience régionale utilise une fenêtre d'étude prédéfinie commençant le
1er juin 2013, ou la date de début du profil si elle est plus récente. Ce choix
borne le coût de cette matrice : il ne représente pas la première date commune
disponible des quatre zones. Le centralisé régional et les clients reçoivent
exactement les mêmes fenêtres de cette période. La date effective figure dans
chaque configuration résolue ; les dates de validation et de test restent celles
du profil. Les expériences PJME seules conservent leur historique configuré.

Les six expériences sont configurées dans `configs/experiments/` et exécutées
par `pulse experiment --experiment N --profile PROFIL`. L'orchestrateur commun
consomme les facteurs versionnés de `configs/experiments.yaml` et les budgets
explicites de `configs/profiles/`. Toutes les graines configurées sont utilisées,
sauf restriction explicite `--seed`.

|N|Facteurs|Sorties|Critère d'interprétation|
|---|---|---|---|
|1|6 modèles dont deux saisonniers|Prédictions identiques en timestamps, erreurs, coûts|Comparer aussi au meilleur naïf ; préciser la fenêtre ARIMA|
|2|L=24/48/168, H=16/32|Pertes, checkpoints, métriques|Conserver le manifeste commun et examiner variabilité entre graines|
|3|Centralisé, FedAvg, FedProx|Tours de validation, erreurs finales, mises à jour|Rapporter les budgets sans présumer leur égalité|
|4|IID approximatif, temporel, régional ; mu=0/.001/.01/.1 ; scalers communs/locaux|Diagnostics clients, matrices de distances, métriques régionales|Séparer hétérogénéité temporelle et régions, étudier le changement de coordonnées|
|5|XGBoost sélectionné sur validation|SHAP global/local/dépendance/période|Aucune interprétation causale|
|6|Artefacts terminés des expériences précédentes|summary.csv, figures précision/coût|Ne pas mélanger protocoles, profils ou références de normalisation|

Les expériences 1–5 effectuent les calculs ; la sixième agrège les coûts déjà
mesurés, sans réentraîner arbitrairement. Les tableaux détaillés sont dans les
dossiers de résultats et le document calculé `computed_findings.md`.

Une campagne full/gpu doit être exécutée avant de rédiger une conclusion sur
RQ1–RQ5. Une campagne smoke, même répétée trois fois, ne devient pas une preuve
scientifique. Un profil cpu-light est un protocole à budget réduit, identifié
séparément. Les échecs restent dans status.json et ne sont jamais convertis en
valeurs de performance.
