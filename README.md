# Projet pMCMC

Pour une description plus détaillée du modèle SEIR, voir ce [github](https://institutefordiseasemodeling.github.io/Documentation/general/model-seir.html). 

### Prérequis

Cette implémentation du pMCMC pour l'estimation de l'effective contact rate
repose sur le package ```particles``` de N. Chopin disponible sur github 
[ici](https://github.com/nchopin/particles). 

### Fichiers

__Fichiers principaux__
- ```donnees_simulees.ipynb``` permet de simuler des données manuellement (les données 
sont enregistées dans le fichier 'data_with_beta.csv' qui est écrasé à chaque instruction).
- ```data_with_beta.csv``` contient un tableau à trois colonnes : date, incidence et 
log beta associé.
- ```main.py``` permet d'exécuter l'algorithme de pMCMC pour le modèle SEIR et d'afficher
les résultats.

Les autres fichiers sont des ressources utiles ou des classes utilisées par les fichiers 
principaux. Pour plus d'informations sur ces dernières, se référer directement à la 
documentation dans le code.

### Génération des données

Pour simuler les données, exécuter le notebook donnees_simulees. 

### Exécution de l'algorithme pMCMC