# Pulse Predict — méthodologie de reconstruction

## Statut et motivation

Ce dépôt reconstitue le périmètre documenté d'un projet de stage : prévision de
charge électrique, comparaison XGBoost/LSTM, apprentissage centralisé/fédéré,
FedAvg/FedProx et explicabilité SHAP. Les sources originales ont été perdues.
Le code, les jeux d'essais publics, les protocoles et les extensions présents ici
sont nouveaux. Une expérience exécutée aujourd'hui ne prouve aucun résultat
historique du stage. Aucune mise en production ni collaboration industrielle
effective entre clients ne découle d'une simulation locale.

La charge horaire combine cycles quotidiens, hebdomadaires, saisonniers et
changements de régime. Une méthode utile doit améliorer des références simples
sur des périodes futures, avec un coût mesuré et des limites explicites. La
fédération ajoute une question de répartition des données et d'optimisation,
pas seulement un algorithme différent.

## Questions de recherche et artefacts

|Question|Expériences|Mesures et tables|Discussion attendue|
|---|---|---|---|
|RQ1 : précision|1|MAE/RMSE/sMAPE, prédictions communes, summary.csv|Gain par rapport aux naïfs, limites ARIMA|
|RQ2 : centralisé/fédéré|3|Erreurs, exemples traités, mises à jour, durées|Différences de budgets et d'accès|
|RQ3 : hétérogénéité/FedProx|4|Distances clients, erreurs régionales, sensibilité mu|Dérive locale, représentation, normalisation|
|RQ4 : variables importantes|5|SHAP, classements par période, cas locaux|Association prédictive et corrélations|
|RQ5 : compromis|6|Durée, octets de paramètres, erreur/coût|Localité simulée, explicabilité, limites système|

Les tableaux numériques et figures proviennent exclusivement d'artefacts
exécutés. Les résultats smoke sont des contrôles fonctionnels. Tant qu'une
campagne scientifique n'a pas été exécutée, la discussion associée reste une
question ouverte, sans classement de performance affirmé.

## Données et provenance

Source principale : [Hourly Energy Consumption, Rob Mulla/Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).
Référence opérateur : [PJM Data Miner 2](https://www.pjm.com/markets-and-operations/etools/data-miner-2).
Les fichiers régionaux associent un Datetime naïf et une colonne telle que
PJME_MW. Le suffixe et les métadonnées de provenance déclarent une puissance en
MW ; il ne s'agit pas d'une énergie horaire en MWh à confondre avec la cible.
Le contenu numérique ne permet pas de certifier indépendamment les unités.
La licence doit être vérifiée auprès de la source avant redistribution.

Le chargeur vérifie les dates, heures alignées, colonnes cibles, nombres finis,
valeurs négatives et couverture. Il trie les observations et calcule une
empreinte SHA256 de chaque fichier. Les fichiers régionaux ne sont jamais
additionnés implicitement. PJM_Load historique n'est pas supposé représenter
le même périmètre que PJME ou AEP.

Les dates sans fuseau ne justifient pas une conversion UTC automatique.
Des doublons à proximité du changement d'heure et des heures manquantes peuvent
refléter le calendrier, la convention d'heure de fin, un défaut d'export ou une
combinaison de ces causes. Le protocole civil conserve les CSV, masque toutes
les occurrences des timestamps dupliqués et matérialise les absences par NaN.
Il exclut les fenêtres traversant ces anomalies et compte les exclusions.
Il ne prétend pas résoudre la correspondance entre ces heures et des instants
UTC. Les valeurs extrêmes sont signalées par IQR et conservées : une pointe de
charge réelle ne doit pas être supprimée parce qu'elle est rare.

## Formulation et absence de fuite

À l'origine t−1, on prédit y_t, la charge de l'heure civile suivante. Les
informations disponibles comprennent les charges observées jusqu'à t−1 et le
calendrier connu pour t. L'évaluation est séquentielle à une heure : après
l'arrivée de y_t, cette observation peut entrer dans la prévision de t+1.
Ce protocole ne constitue pas une prévision simultanée de plusieurs mois.

Les variables XGBoost comprennent heure, jour de semaine, mois, week-end,
sinus/cosinus pour les cycles, retards 1/24/48/168, et moyenne/écart-type sur
24/168 heures. Une moyenne de largeur w au timestamp t est
`mean(y[t-w:t])` ; le décalage d'une heure précède tout rolling. Un simple
rolling incluant y_t serait une fuite directe.

Les splits scientifiques sont chronologiques : avant 2017 pour l'apprentissage,
2017 pour la validation et du 1er janvier au 3 août 2018 exclus pour le test.
Les fenêtres de validation peuvent utiliser le passé d'entraînement ; leurs
cibles ne sont jamais ajoutées à l'ensemble d'optimisation. La disponibilité
commune des historiques définit les timestamps admissibles avant le fit.
Toutes les longueurs étudiées jusqu'à 168 utilisent ce même support commun.
Une différence de prédictions ne peut jamais modifier après coup le manifeste.

Le LSTM utilise une standardisation z=(y−m)/s. Les moments m,s sont calculés
sur les heures uniques effectivement accessibles dans l'entraînement, sans
surpondérer artificiellement les fenêtres qui se chevauchent. La transformation
inverse est y=z·s+m. Les modèles, scalers et schémas sont enregistrés ensemble.
La sélection, le stopping et les réglages utilisent uniquement la validation.

## Références de prévision

Le naïf prédit y[t−1]. Le saisonnier journalier prédit y[t−24] et l'hebdomadaire
y[t−168]. Ils ont un coût d'entraînement nul mais constituent des références
fortes sur des séries périodiques. Une amélioration sur un seul naïf ne suffit
pas à établir la supériorité générale d'un modèle complexe.

ARIMA représente la série différenciée par des termes autorégressifs et une
moyenne mobile : φ(B)(1−B)^d y_t = c + θ(B)ε_t. La grille bornée est
(1,0,0), (2,0,0), (1,1,1), sélectionnée par AIC parmi les fits convergés. Le
fit utilise les 90 derniers jours disponibles de l'entraînement. Le filtre
traite ensuite validation/test avec paramètres gelés ; la prévision à un pas
utilise les observations précédentes, sans réestimer les paramètres sur le test.
Les valeurs absentes restent manquantes dans le modèle espace d'état.
Ce choix rend ARIMA calculable sur une machine personnelle, mais sa fenêtre
d'apprentissage diffère des autres modèles et doit apparaître dans l'analyse.

## XGBoost

Une prédiction est une somme d'arbres : ŷ_i=Σ_k f_k(x_i). À chaque étape, le
nouvel arbre corrige les erreurs sous une approximation de Taylor de la perte.
L'objectif combine perte quadratique et régularisation de la complexité :
`Σ_i l(y_i,ŷ_i) + Σ_k [γ T_k + λ/2 Σ_j w_kj²]`. Les gradients et Hessiennes
guident les partitions et valeurs des feuilles. La profondeur contrôle les
interactions, le learning rate atténue chaque ajout et le nombre d'arbres
détermine la capacité totale.

Une petite grille de profondeurs est évaluée sur validation. L'arrêt anticipé
retient la meilleure itération ; les prédictions et explications utilisent
cette plage d'arbres, pas les arbres postérieurs à l'optimum. Toutes les
configurations candidates et leurs scores sont enregistrées. L'importance par
gain est descriptive ; elle n'est pas interchangeable avec une attribution SHAP.
Les ensembles d'arbres XGBoost ne sont pas soumis à une moyenne de poids FedAvg.

## LSTM

Pour l'entrée x_t, l'état caché h et la mémoire c, un LSTM calcule :

```text
i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)
f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)
g_t = tanh(W_g [h_(t-1), x_t] + b_g)
c_t = f_t * c_(t-1) + i_t * g_t
o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)
h_t = o_t * tanh(c_t)
```

La porte d'oubli module la mémoire précédente, la porte d'entrée contrôle
l'écriture et la porte de sortie expose une partie de la mémoire. Ces mécanismes
facilitent le transport du gradient, sans garantir qu'un cycle de 168 heures sera
appris sur une séquence de 24 heures. La sensibilité à la longueur est donc une
expérience, pas une conclusion présumée.

Le modèle initial est volontairement simple : une couche LSTM et une tête
linéaire sur le dernier état. Les fenêtres de charge ont la forme [B,L,1], les
sorties récurrentes [B,L,H] et la tête renvoie [B]. Adam optimise MSE ou MAE,
avec minibatches, clipping et validation. Le checkpoint retenu correspond au
minimum de perte de validation. Les pertes sont normalisées ; les métriques
finales sont en MW. Centralisé et fédéré sélectionnent leur checkpoint avec
la même métrique de validation en MW : RMSE pour une perte MSE, MAE pour une
perte MAE. Cela évite de favoriser implicitement les petites échelles avec des
normalisations locales. Les courbes d'apprentissage et résidus permettent de
repérer sous-apprentissage, surapprentissage et biais de régime.

## Fédération et hétérogénéité

À un tour r, chaque client k reçoit w^r et optimise sa perte locale sur ses n_k
fenêtres. FedAvg agrège `w^(r+1)=Σ_k (n_k/Σ_j n_j) w_k^(r+1)`. Une époque
locale parcourt les données d'un client ; un tour correspond à un échange et
une agrégation. Les deux notions ne sont pas des unités de coût équivalentes.

FedProx minimise `F_k(w)+(mu/2)||w−w^r||₂²`. Les références w^r sont des copies
détachées, fixes durant toutes les époques du tour. Le gradient du terme ajouté
est mu·(w−w^r). La norme doit être au carré ; une somme des normes non carrées
constituerait un autre objectif. Lorsque mu=0, les mises à jour locales doivent
être identiques à celles de FedAvg sous les mêmes données, minibatches,
initialisations et états d'optimiseur.

Les clients IID approximatifs reçoivent des fenêtres réparties uniformément
avec un seed de partition fixe puis triées chronologiquement. Les historiques
peuvent se chevaucher et les observations restent corrélées. Les partitions
temporelles hétérogènes sont des blocs successifs d'une même série ; elles
simulent des distributions différentes mais pas des entreprises indépendantes.
Le scénario régional utilise des zones distinctes sur une couverture commune.
Les diagnostics sont calculés sur les seules données d'entraînement : moments,
quantiles, profils horaires, distance de Wasserstein en MW et divisée par
l'écart-type global des cibles des clients.

Dans la référence multirégionale, un scaler commun d'entraînement fournit les
mêmes coordonnées à tous les clients et au centralisé. Le partage de statistiques
agrégées est une hypothèse du protocole, non un mécanisme de confidentialité.
Avec des scalers locaux, une même unité normalisée correspond à des MW différents.
Moyenner les paramètres devient une ablation de représentation : il faut
évaluer chaque région après inversion de son propre scaler et discuter les
différences, sans affirmer que cette agrégation est équivalente à la référence.

Le centralisé reçoit l'union des mêmes fenêtres ; le protocole rapporte les
exemples traités, mises à jour, durées et budgets. L'arrêt anticipé centralisé et
les tours fédérés n'impliquent pas une égalité de ressources. Tous les clients
participent à chaque tour. Les simulations ne prouvent pas une résistance aux
pannes réseau, une confidentialité différentielle, une agrégation sécurisée ou
une performance de déploiement réel.

## Mesures, répétitions et coût

MAE=`mean(|ŷ−y|)` mesure une erreur moyenne absolue en MW. RMSE=
`sqrt(mean((ŷ−y)²))` pénalise davantage les grandes erreurs, dans la même unité.
sMAPE=`100 mean(2|ŷ−y|/(|y|+|ŷ|))`, avec contribution zéro si les deux valeurs
sont nulles. Les agrégats régionaux comprennent la moyenne macro des métriques
et les métriques sur observations concaténées ; RMSE pooled est la racine des
MSE pondérées, pas une moyenne naïve des RMSE.

Les graines 42/43/44 partagent périodes, partition et protocole. Moyenne et
écart-type échantillonnal décrivent la variabilité d'optimisation ; les graines
ne rendent pas indépendantes les erreurs horaires. Aucune significativité n'est
affirmée sans test adapté à la dépendance temporelle. Le bootstrap par blocs
reste une extension possible.

Les durées dépendent du matériel et des tâches concurrentes. Les octets simulés
de paramètres sont `2 × clients × tours × taille_des_tenseurs`, pour diffusion
et remontée. Ce calcul exclut métadonnées, sérialisation, évaluations et
infrastructure Ray ; il ne mesure pas un réseau réel. Le temps SHAP doit être
compté séparément. Les profils de calcul et restrictions explicites sont stockés.

## Explicabilité et limites de validité

TreeExplainer répartit l'écart entre prédiction et valeur de référence entre
les variables. L'analyse interventional utilise un fond d'entraînement, jamais
le test pour sélectionner ce fond. Les explications globales agrègent |SHAP| ;
les cas locaux montrent contributions signées et valeur de base. Les dépendances
et classements par période servent à examiner la stabilité.

Les retards et moyennes glissantes sont corrélés ; les attributions peuvent
être partagées de manière instable entre variables redondantes. SHAP ne prouve
pas de causalité. Le choix du fond, du modèle et de l'échantillonnage influence
les valeurs. Les premières explications sont celles de XGBoost, pas du LSTM.

Menaces : un seul ensemble de données historique, calendrier civil incertain,
exclusions proches des transitions, absence de météo, absence d'événements
économiques, décalage entre régions, faible nombre de graines et budgets limités.
Une normalisation commune utilisant toutes les régions présuppose un canal de
statistiques ; une simulation locale ne valide pas une architecture industrielle.
Les suites possibles sont données PJM mieux documentées, validation glissante,
météo disponible à l'origine, intervalles de prévision, test temporel apparié,
confidentialité différentielle et horizon 24 heures comme protocole distinct.

## Références

1. McMahan et al. (2017), [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html).
2. Li et al. (2020), [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127).
3. Chen & Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
4. Hochreiter & Schmidhuber (1997), [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735).
5. Lundberg & Lee (2017), [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
6. [Flower simulation runtime](https://flower.ai/docs/framework/how-to-run-simulations.html), documentation vérifiée pour la version épinglée ; les signatures installées sont la référence technique.
7. [XGBoost 3.0 Python API](https://xgboost.readthedocs.io/en/release_3.0.0/python/python_api.html).
