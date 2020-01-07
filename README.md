# Projet pMCMC

Pour une description plus détaillée du modèle SEIR, voir ce [github](https://institutefordiseasemodeling.github.io/Documentation/general/model-seir.html). 

Cette implémentation est fondée sur l'article ```MainDocument.pdf```.

### Prérequis

Cette implémentation du pMCMC pour l'estimation de l'effective contact rate
repose sur le package ```particles``` de N. Chopin disponible sur github 
[ici](https://github.com/nchopin/particles). 

### Fichiers

__Fichiers principaux__
- ```donnees_simulees.ipynb``` permet de simuler des données manuellement (les données 
sont enregistées dans le fichier 'data_with_beta.csv' qui est écrasé à chaque exécution).
- ```data_with_beta.csv``` contient un tableau à trois colonnes : date, incidence et 
log beta associé.
- ```main.py``` permet d'exécuter l'algorithme de pMCMC pour le modèle SEIR et d'afficher
les résultats.
- ```raw_data.csv``` contient les données utilisées dans l'article.

Les autres fichiers sont des ressources utiles ou des classes utilisées par les fichiers 
principaux. Pour plus d'informations sur ces dernières, se référer directement à la 
documentation dans le code.

### Génération des données

Pour simuler les données, exécuter le notebook ```donnees_simulees.ipynb```. 

### Exécution de l'algorithme pMCMC

Il suffit d'exécuter le script ```main.py```.

Pour changer le nombre de particules ou le nombre d'itérations, il suffit de modifier 
respectivement dans  le fichier ```main.py``` les valeurs des variables ```N_PART``` 
et ```N_ITER```.

Pour changer les valeurs des coefficients ou les valeurs initiales du système d'équations
différentielles, il suffit de modifier respectivement les valeurs par défaut de ```coefs``` et de ```init```
dans le fichier ```my_state_space_models.py```.

