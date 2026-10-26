# Résultats et discussion — statut

Ce fichier ne contient aucun résultat historique reconstitué de mémoire.
Les chiffres actuels doivent être consultés dans les artefacts calculés et leur
`computed_findings.md`, avec configuration, seed, profil et empreinte des données.
`docs/validation.md` indique les parcours réellement vérifiés.

La matrice fonctionnelle finale comprend 120 exécutions réussies sur données
réelles. Son récapitulatif est `results/validated/computed_findings.md`, avec les
tables `summary.csv` et `cost_summary.csv`. Ces valeurs démontrent le parcours
logiciel ; elles ne répondent pas encore scientifiquement aux questions ci-dessous.

|Question|Conclusion scientifique actuelle|Éléments nécessaires|
|---|---|---|
|RQ1|Non établie par une validation fonctionnelle|Expérience 1 complète, mêmes timestamps, trois graines|
|RQ2|Non établie par trois tours smoke|Expérience 3 Linux/Ray, budgets documentés|
|RQ3|Aucune supériorité FedProx présumée|Expérience 4, distributions, mu et ablations de normalisation|
|RQ4|Les explications d'un modèle smoke ne caractérisent pas le phénomène général|Expérience 5 sur modèle scientifique et plusieurs périodes|
|RQ5|Les coûts du transport en mémoire ne mesurent pas Ray ou un déploiement|Expérience 6 sur protocoles identifiés et matériel documenté|

Lors de la rédaction finale, discuter les gains absolus en MW, leur dispersion
entre graines, les erreurs de pointe, la différence de fenêtre ARIMA et les
coûts d'optimisation. Un FedProx plus lent ou moins précis sur certains clients
est un résultat possible, pas un échec à dissimuler. Examiner la sensibilité
des attributions sous retards corrélés et les différences de représentation
entre scalers locaux et communs. Toute conclusion doit citer un chemin
d'artefact précis ; aucune significativité statistique n'est présumée.
