# Validation locale et limites d'exécution

## Environnement

Le projet a été créé dans le dossier `pulse` contenant les CSV d'origine.
Un environnement Windows Python 3.11 isolé a été installé ; `pip check` a
réussi. Les dépendances directes sont épinglées et l'instantané Windows est
dans `requirements-lock-windows.txt`. Torch est ici installé sans CUDA ; la
présence matérielle d'une GTX 1650 ne signifie pas qu'un entraînement GPU ait
été validé.

## Contrôles exécutés

- Première suite centralisée : 16 tests réussis, incluant l'inférence inédite et
  la causalité du filtrage ARIMA.
- Suite incluant Flower : 17 tests réussis et un test Linux/Ray explicitement
  ignoré sur Windows. L'adaptateur fonctionnel a nécessité l'initialisation de
  l'identité d'exécution demandée par Flower 1.40 ; la correction a été testée.
- Rechargement des checkpoints LSTM et XGBoost et inférence sur le CSV PJME
  complet : prédictions finies, incluant l'heure suivant la dernière observation.
- SHAP exécuté sur le profil court PJME : importance, résumé, dépendance et
  explication locale ; valeurs et comparaison par période sauvegardées.
- Quatre notebooks validés contre leur schéma nbformat. Cette validation de
  structure n'est pas une exécution intégrale de leurs cellules.

Le parcours final est automatisé par :

```shell
python scripts/validate_project.py --matrix
```

Il utilise le même interpréteur pour pip, pytest et les commandes CLI. Les
journaux et codes de retour réels sont conservés dans
`results/validated/validation.json` et `command_*.log`. Il compare les hashes des
CSV avant/après et vérifie indépendamment chaque manifeste de prédictions.
La validation finale a terminé avec succès : les quatre commandes ont renvoyé
le code 0 et les 120 entraînements/évaluations fonctionnels sur trois graines
ont produit leurs artefacts. Tous les manifests de prédictions ont été revérifiés
et les empreintes des CSV d'origine sont identiques avant/après.

La vérification supplémentaire suivante a également réussi :

```shell
python scripts/check_federated_equivalence.py --results results/validated
```

Elle confirme une égalité exacte des prédictions FedAvg/FedProx à mu=0 sur
12 paires de protocoles et 30 comparaisons régionales. La preuve détaillée est
dans `results/validated/fedprox_zero_real_checks.json`. Ce contrôle fonctionnel
ne prouve pas l'intérêt d'un mu positif.

`results/validated/source_snapshot.zip` archive les sources et configurations
utilisées. Son empreinte de sources a été vérifiée contre celles de toutes les
exécutions finales : elles correspondent. Les tableaux, coûts et figures
calculés sont accompagnés de `computed_findings.md`, sans conclusion scientifique
tirée du profil smoke.

Les essais exploratoires précédents restent dans `results/smoke`,
`results/smoke-matrix`, `results/shap-check` et les dossiers d'inférence. Pour
examiner le code final, privilégier les artefacts `results/validated`.

## Blocages et vérifications non effectuées

- **Linux/Ray** : la seule distribution WSL initialement détectée est
  `docker-desktop`. Aucun Ubuntu de développement n'est configuré et le moteur
  Docker Linux est arrêté. La commande Ray depuis Windows a été vérifiée : elle
  s'arrête explicitement et ne change pas de backend. Le test Ray demeure
  ignoré localement ; son job CI Linux est configuré mais n'a pas été exécuté
  sur GitHub pendant cette reconstruction.
- **CUDA** : le profil GPU doit être exécuté dans un environnement Torch/CUDA
  compatible. Son refus explicite en absence de CUDA est testé ; ses performances
  GPU ne sont pas validées ici.
- **Campagne scientifique** : les profils full/gpu n'ont pas été exécutés.
  Les trois tours du profil smoke n'établissent ni convergence ni supériorité.
- Des avertissements de dépréciation de dépendances Typer/Click et
  Matplotlib/Pyparsing apparaissent ; ils n'ont pas empêché les tests réussis.

Le backend inprocess est un transport synchrone de diagnostic utilisant les
vraies applications et stratégies Flower. Il ne valide ni les processus Ray,
ni la communication réseau, ni la confidentialité d'un déploiement réel. Son
adaptateur d'identité utilise un composant interne de Flower 1.40 ; cette
dépendance est confinée au transport de test et protégée par le test
d'équivalence de bout en bout. Le parcours scientifique utilise le runtime
officiel qui initialise lui-même son identité.
