# Préparation d'entretien — Orange Innovation

## Positionnement honnête

« Pulse Predict est une reconstruction du périmètre documenté de mon travail
de stage : prévision de charge, XGBoost/LSTM, comparaison centralisé/fédéré et
SHAP. Le dépôt original a été perdu. Je présente ici une nouvelle implémentation
reproductible sur données publiques PJM, sans prétendre retrouver ses résultats
exacts. Je distingue les activités historiques attestées, les choix de
reconstruction et les extensions nouvelles. »

Ne pas annoncer de gain chiffré sans ouvrir le résultat calculé correspondant.
Une simulation de quelques tours prouve un fonctionnement, pas la convergence
ou la supériorité d'une méthode.

## Présentation en trente secondes

« Je reconstruis une chaîne de prévision de charge électrique à une heure.
Je compare des références naïves et ARIMA à XGBoost et à un LSTM, puis le même
réseau en apprentissage fédéré avec FedAvg et FedProx. Mon attention porte sur
l'absence de fuite temporelle, l'équité des comparaisons et l'explication des
prévisions avec SHAP. Chaque résultat est relié aux données, à sa configuration
et à sa graine ; les limites de la simulation sont explicites. »

## Présentation technique en deux minutes

« Le problème consiste à estimer la charge de la prochaine heure avec les seules
observations déjà disponibles. Les CSV PJM présentent des heures manquantes et
ambiguës. Je commence par les auditer, conserver les sources et exclure les
fenêtres dont la chronologie n'est pas exploitable, plutôt qu'inventer des
valeurs. Le découpage est chronologique : entraînement, validation et test futur.

Pour XGBoost, je construis des retards de charge, des moyennes historiques et des
variables calendaires. Le LSTM reçoit une séquence de charges normalisées sur
l'entraînement uniquement. Je vérifie que tous les modèles produisent leurs
prévisions sur exactement les mêmes timestamps de test, puis mesure MAE et RMSE
en MW. Des naïfs journalier et hebdomadaire empêchent de surestimer l'intérêt
d'un modèle complexe.

En fédéré, plusieurs clients simulés entraînent localement le même LSTM.
FedAvg moyenne leurs paramètres selon leurs effectifs. FedProx ajoute une
pénalité quadratique limitant l'écart au modèle global. Je teste des répartitions
IID approximatives, temporelles hétérogènes et régionales, en quantifiant leurs
différences. Une normalisation commune sert de référence ; les normalisations
locales sont une ablation distincte. Enfin, SHAP explique les prédictions du
modèle XGBoost sans implication causale. Les coûts et résultats sont enregistrés
sur trois graines ; les vérifications smoke restent séparées de la recherche. »

## Présentation d'architecture en cinq minutes

1. **Provenance et contrat d'entrée.** Les données sont des charges régionales
   publiques exprimées en MW. Le chargeur découvre les colonnes, contrôle dates
   et valeurs, calcule un hash et produit un audit. Les différentes régions ne
   représentent pas une seule mesure interchangeable.
2. **Chronologie.** Je fixe les frontières et le support historique requis.
   Les doublons ambigus et absences sont masqués dans une grille d'heures
   civiles. La convention UTC exacte reste une limite tant qu'elle n'est pas
   justifiée par les métadonnées. Un manifeste détermine les observations
   comparables avant l'entraînement.
3. **Représentations.** XGBoost reçoit un tableau de variables disponibles à
   l'origine. Le LSTM reçoit un tenseur [batch, longueur, variables], ici une
   variable de charge. Les transformations sont ajustées sur l'entraînement.
4. **Optimisation.** Les baselines donnent un point de comparaison. XGBoost
   est réglé par une petite recherche sur validation ; le LSTM utilise Adam,
   minibatches, clipping et arrêt anticipé. Je sauvegarde le meilleur modèle
   avec ses transformations pour garantir une inférence cohérente.
5. **Fédération.** Le serveur Flower diffuse les poids, chaque client optimise
   sa perte locale et le serveur agrège selon les effectifs. FedProx utilise
   une copie détachée du modèle global. Les clients et budgets sont explicites.
   Les messages transmettent des paramètres, pas les observations brutes, mais
   cela ne constitue pas une garantie formelle de confidentialité.
6. **Évaluation et interprétation.** Je dénormalise, vérifie l'égalité des
   timestamps, calcule les métriques globales/régionales et les coûts. Les
   trois graines décrivent une variabilité d'entraînement. Les résidus,
   trajectoires de validation et SHAP aident à comprendre les résultats.
7. **Preuves.** Chaque exécution a une configuration résolue, des hashes,
   versions, prédictions et checkpoints. Les tests attaquent notamment les
   fuites temporelles et l'équivalence FedAvg/FedProx à mu nul. Je termine en
   exposant ce qui a été réellement exécuté et les expériences encore ouvertes.

## Explications pas à pas

### XGBoost

On commence par une prédiction de référence. À chaque itération, on calcule les
gradients de la perte par rapport aux prédictions actuelles et leurs dérivées
secondes. Un nouvel arbre recherche des séparations utiles entre valeurs de
variables, sous régularisation. Les feuilles apportent des corrections, ajoutées
avec un learning rate. On répète jusqu'à la limite ou à l'arrêt anticipé sur
validation. L'ensemble final est la somme des arbres retenus. Une profondeur
plus grande peut capturer davantage d'interactions mais aussi surapprendre.

### LSTM : portes et mémoire

À chaque heure d'une fenêtre, le réseau combine l'entrée avec l'état précédent.
La porte d'oubli choisit quelle fraction de mémoire conserver. La porte d'entrée
choisit quelle fraction d'une nouvelle mémoire candidate écrire. Leur
combinaison donne la mémoire courante. La porte de sortie détermine la partie
exposée dans l'état caché. La tête linéaire transforme le dernier état en charge
normalisée future. La rétropropagation traverse les heures et ajuste les poids.
Les équations détaillées figurent dans methodology.md. Il ne faut pas confondre
mémoire apprise et stockage explicite de toutes les observations passées.

### Fenêtres et fuite temporelle

Pour prévoir y_t avec L=24, je prends y[t−24], …, y[t−1] comme entrée et y_t
comme cible. Pour t+1, la fenêtre glisse d'un cran. Les fenêtres peuvent partager
23 observations, ce qui explique leur forte dépendance. Un split aléatoire des
lignes peut donc rendre le test artificiellement proche de l'entraînement.
Une moyenne de largeur 24 doit être calculée après shift(1). Le scaler ne doit
pas connaître la moyenne du test. Les tests modifient les valeurs futures et
vérifient que les features et transformations passées restent identiques.

### FedAvg, FedProx et non-IID

Le modèle initial est commun. À chaque tour, les clients partent des mêmes
poids globaux, font leurs époques locales, puis renvoient leurs poids. FedAvg
calcule une moyenne pondérée par le nombre d'exemples. FedProx optimise la
prédiction plus `(mu/2) × somme des carrés des écarts au modèle global`. Un mu
trop grand peut empêcher l'adaptation ; un mu trop faible peut laisser diverger
les mises à jour. Non-IID signifie que les distributions diffèrent : niveaux,
saisonnalité, variance ou périodes. Cela peut rendre les directions locales
incompatibles. Le terme proximal est une réponse possible, pas une garantie
de supériorité sur toute distribution.

### SHAP et erreurs

Une explication SHAP décompose une prédiction autour d'une valeur de référence.
Une contribution positive augmente la prévision par rapport à cette référence ;
une contribution négative la diminue. Une importance globale moyenne les
valeurs absolues. Avec des retards corrélés, plusieurs variables peuvent se
partager une même information et leurs attributions changer. Ce n'est pas une
analyse causale. MAE mesure l'amplitude moyenne de l'erreur ; RMSE accentue les
grosses erreurs. Toutes deux restent en MW et doivent être interprétées par
rapport au niveau de charge et aux erreurs des baselines.

## Questions techniques et réponses

### 1. Pourquoi prédire à une heure plutôt qu'à vingt-quatre heures ?
L'horizon fixe les informations disponibles. À une heure, la dernière charge
observée est très informative. À vingt-quatre heures, utiliser les vingt-trois
observations intermédiaires serait une fuite. Je préfère valider un protocole
clair à une heure ; un horizon 24 heures exige un jeu de features et une
évaluation séparés.

### 2. Pourquoi ne pas utiliser un split aléatoire ?
Il brise la chronologie et mélange des fenêtres presque identiques. La
performance peut refléter l'accès à des régimes futurs plutôt qu'une capacité
de généralisation. Je réserve des périodes futures entières et conserve la
validation avant le test.

### 3. Est-ce une fuite d'utiliser les dernières heures d'entraînement pour valider ?
Non, si ces heures sont déjà observées à l'origine. Une frontière de split porte
sur les cibles et les données utilisées pour ajuster les paramètres ; elle ne
doit pas supprimer un historique légitimement disponible. La cible de
validation ne devient jamais une cible d'apprentissage.

### 4. Pourquoi shift avant rolling ?
Un rolling non décalé inclut la cible courante. À l'origine t−1, cette cible est
inconnue. shift(1) déplace la dernière observation admissible et permet une
moyenne portant uniquement sur les heures antérieures.

### 5. Comment vérifier l'absence de fuite du scaler ?
Je calcule ses moments avec les seules heures d'entraînement. Un test remplace
toutes les valeurs validation/test par des valeurs extrêmes et vérifie que les
moments restent identiques. Ce test est exécuté pour la normalisation commune
et locale, y compris plusieurs régions.

### 6. Que faire des heures dupliquées lors d'un changement d'heure ?
Ne pas choisir arbitrairement une ligne ou supposer un fuseau. Je conserve le
CSV, signale les doublons et masque les heures ambiguës dans ce protocole.
Leur résolution exacte nécessite une convention de provenance vérifiée,
notamment heure de début/fin et fuseau. Les exclusions sont quantifiées.

### 7. Pourquoi conserver les pointes de charge ?
Une pointe peut être précisément ce que le modèle doit prévoir. La supprimer
par un seuil purement statistique réduit artificiellement la difficulté.
Je signale les anomalies, vérifie les unités et ne corrige qu'avec une
justification de qualité des données.

### 8. Quelles différences entre XGBoost et LSTM ?
XGBoost apprend des règles non linéaires sur des features historiques explicites.
Le LSTM apprend une représentation séquentielle avec des poids partagés au
fil du temps. Le premier peut être très performant et économique sur données
tabulaires ; le second apporte une structure séquentielle mais nécessite
normalisation, optimisation et choix de fenêtre.

### 9. Pourquoi un LSTM plutôt qu'un Transformer ?
Le but initial est une reconstruction défendable, compatible avec une machine
personnelle et comparable au périmètre documenté. Une architecture plus coûteuse
ajouterait des facteurs sans assurer un bénéfice. Un Transformer serait une
extension après établissement de baselines solides.

### 10. Quelles formes de tenseurs utilisez-vous ?
L'entrée est [B,L,1], la sortie LSTM [B,L,H]. Je sélectionne le dernier état
[B,H], puis une tête linéaire produit [B,1], converti en [B] pour la perte.
Une incohérence de squeeze peut entraîner un broadcasting accidentel de la
perte ; les tests vérifient ces dimensions.

### 11. Pourquoi clipper les gradients ?
Les réseaux récurrents peuvent accumuler des gradients très grands. Le clipping
limite leur norme pour stabiliser une mise à jour, sans résoudre toutes les
causes d'instabilité. Son seuil est versionné et identique dans les comparaisons.

### 12. Comment choisissez-vous les hyperparamètres ?
Par une recherche bornée et une validation chronologique. Je conserve les
candidats, scores et paramètres retenus. Le test ne sert pas à choisir la
profondeur ou l'itération XGBoost. Les sensibilités LSTM sont rapportées comme
expériences, sans transformer le meilleur test en modèle pré-sélectionné.

### 13. Pourquoi inclure un naïf saisonnier ?
La charge présente des cycles forts. Le dernier jour ou la dernière semaine
peut déjà offrir une bonne prévision à coût presque nul. Une comparaison
scientifique doit établir ce que les modèles appris apportent au-delà de ce
repère, sur les mêmes heures.

### 14. Votre ARIMA reçoit-il exactement les mêmes données d'entraînement ?
Non : son estimation est bornée aux 90 derniers jours d'entraînement pour
maîtriser le calcul. Son test est commun aux autres méthodes et ses paramètres
restent gelés, mais sa fenêtre d'apprentissage diffère. Je le signale au lieu
de présenter une comparaison parfaitement contrôlée sur tous les axes.

### 15. Quelle différence entre un tour et une époque locale ?
Un tour comprend la diffusion, l'optimisation des clients et l'agrégation.
Une époque locale parcourt une fois les exemples d'un client. Plusieurs
époques par tour augmentent le calcul local et peuvent accroître la dérive
des modèles entre distributions différentes.

### 16. Pourquoi pondérer FedAvg par les effectifs ?
Cela correspond à un objectif empirique global pondéré par observations.
Une moyenne uniforme des clients favoriserait proportionnellement les petits
clients et représenterait un autre objectif. L'équité entre régions doit être
discutée aussi avec des métriques macro, pas seulement pooled.

### 17. Comment vérifiez-vous FedProx ?
Je vérifie que le gradient du terme est mu·(w−w_global), que les références
sont détachées et que la somme porte sur les paramètres entraînables. À mu=0,
j'exige les mêmes paramètres finaux que FedAvg sous un ordre de minibatches,
une initialisation, des données et un optimiseur identiques.

### 18. Pourquoi les scalers locaux compliquent-ils FedAvg ?
Un même nombre normalisé peut désigner des charges différentes selon le client.
Le réseau partagé travaille alors dans des coordonnées physiques différentes.
Je prends un scaler commun d'entraînement comme référence et étudie les
scalers locaux séparément. Les erreurs finales sont toujours dénormalisées
avec le scaler applicable à la région.

### 19. Vos clients sont-ils vraiment IID ?
Seulement approximativement : la répartition équilibre les fenêtres, mais
leurs historiques se chevauchent et restent corrélés. Je quantifie niveaux,
variances, profils et distances de distributions. Les blocs chronologiques
sont explicitement des partitions temporelles hétérogènes simulées.

### 20. La fédération garantit-elle la confidentialité ?
Non. Elle évite l'envoi des observations brutes dans le protocole, mais les
paramètres peuvent révéler des informations. Ici tous les clients sont simulés
dans un environnement de confiance. Une garantie formelle demanderait des
mécanismes supplémentaires et leur analyse, par exemple confidentialité
différentielle et agrégation sécurisée.

### 21. Peut-on fédérer XGBoost en moyennant ses arbres ?
Pas avec le FedAvg de réseaux utilisé ici. Les arbres n'ont pas des paramètres
alignés de la même façon. Des méthodes fédérées spécifiques existent mais
nécessiteraient une implémentation et une justification propres. XGBoost reste
la référence centralisée de ce projet.

### 22. Que signifie une RMSE plus grande que la MAE ?
C'est attendu : les carrés donnent plus de poids aux grandes erreurs. Un écart
important suggère une queue d'erreurs ou des pointes mal prédites. J'examine
alors résidus et périodes, sans conclure à partir d'une seule moyenne.

### 23. Trois graines suffisent-elles à prouver une supériorité ?
Non. Elles décrivent la variabilité de l'optimisation à données fixées. Les
erreurs horaires sont dépendantes, et les périodes peuvent être particulières.
Une affirmation de significativité nécessiterait un test temporel justifié,
par exemple une procédure par blocs ou un test apparié adapté.

### 24. Que peut-on conclure d'une figure SHAP ?
On peut expliquer comment le modèle distribue une prédiction par rapport à
une référence. On ne prouve pas qu'une variable cause physiquement la charge.
Je regarde le fond d'entraînement, les corrélations entre retards, les signes
locaux et la stabilité des classements d'une période à l'autre.

### 25. Comment comparez-vous les coûts ?
Je rapporte temps d'entraînement, nombre de mises à jour, exemples traités,
tours et époques locales. L'estimation des octets de paramètres exclut les
surcharges réseau et n'est pas une mesure de débit réel. Le transport en
mémoire smoke ne permet pas de conclure sur Ray ou un système distribué.

### 26. Comment montrer que le code fonctionne réellement ?
J'ouvre les commandes exécutées, résultats de tests, status.json, configurations,
manifeste et prédictions d'une exécution. Je recharge un checkpoint et prédis
sur un fichier inédit. Je distingue les tests synthétiques de CI, les essais
courts sur PJM et une campagne complète, et annonce tout blocage restant.

### 27. Quelles erreurs d'implémentation craignez-vous le plus ?
Rolling incluant la cible, scaler ajusté sur tout le jeu, fenêtres décalées,
splits aléatoires, métriques en unités normalisées, mauvais poids d'agrégation,
référence FedProx non détachée et sélection sur test. Les tests ciblent ces
erreurs plutôt que de vérifier seulement que les scripts ne plantent pas.

### 28. Quelles sont les prochaines étapes de recherche ?
Mieux documenter le calendrier source, enrichir avec météo réellement disponible,
évaluer plusieurs origines temporelles, produire des intervalles de prévision
et ajouter une analyse statistique tenant compte de la dépendance. Une
prévision à vingt-quatre heures et une protection formelle des clients sont
des extensions distinctes, avec nouveaux protocoles et coûts.
