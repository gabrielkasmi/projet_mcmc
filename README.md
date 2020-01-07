# Projet pMCMC

Pour une description plus détaillée du modèle SEIR, voir ce [github](https://institutefordiseasemodeling.github.io/Documentation/general/model-seir.html). 

### Prérequis

Cette implémentation du pMCMC pour l'estimation de l'effective contact rate
repose sur le package ```particles``` de N. Chopin disponible sur github 
[ici](https://github.com/nchopin/particles). 

### Fichiers


# Readme 

Ce dossier contient les fichiers nécessaires à la réplications des résultats du projet du cours de MCMC-SMC.


### Fichiers
- Le fichier 'raw_data.csv' correspond aux données originales de l'article. 
- Le fichier 'data_with_beta.csv' contient des données simulées et une trajectoire de beta
- Le fichier 'donnees_simulees.ipnyb' permet de simuler des données manuellement (les données sont enregistées
dans le fichier 'data_with_beta.csv' qui est écrasé à chaque instruction).


### Instructions

Pour simuler les données, exécuter le notebook donnees_simulees. 

Pour répliquer les résultats du sampler, 