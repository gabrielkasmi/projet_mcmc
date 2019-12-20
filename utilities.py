# coding: utf-8

# Contient toutes les fonctions nécessaires à la génération des données


# Librairies
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import random, math
from scipy.stats import norm
import sdeint
import pandas as pd

# Fonctions

# Plot d'une simulation, des trajectoires de S, E, I, R, du BM
# et de la valeur de beta = exp(x) associée.

# Initialisation des subplots

def plot_realisation(result, t, kwargs_plt_1, kwargs_plt_2, kwargs_plt_3, size):

    """
    Affiche trois plots les un sous les autres représentant la trajectoire simulée
    l'évolution du BM (donc log-beta) et l'évolution de beta en fonction du temps.

    Ne renvoie rien.

    Arguments:
    - result : un np.array contenant les résultats de la SDE.
    - t : un np.array contenant le temps.
    - kwargs_plt_1 : un dictionnaire contenant des paramètres d'affichage customisés pour
    le premier plot
    - kwargs_plt_2 : un dictionnaire contenant des paramètres d'affichage customisés pour
    le deuxième plot
    - kwargs_plt_3 : un dictionnaire contenant des paramètres d'affichage customisés pour
    le troisième plot
    - size : un tuple (x,y) permettant de renseigner la taille désirée pour l'affichage.
    """

    fig, ((ax1, ax2, ax3)) = plt.subplots(3, sharex=True, sharey=False)

    # Plot des variables d'état du système
    ax1.plot(t, result[:,0], label = "S(t)");
    ax1.plot(t, result[:,1], label = "E(t)");
    ax1.plot(t, result[:,2], label = "I(t)");
    ax1.plot(t, result[:,3], label = "R(t)", **kwargs_plt_1);
    ax1.grid();
    ax1.legend();
    ax1.set_title("Evolution des variables S, E, I et R");
    ax1.set_xlabel("Temps (en semaines)");
    ax1.set_ylabel("Proportions");

    # Plot de la partie stochastique
    ax2.plot(t, result[:,4], label = "log_beta_t", c = "brown", **kwargs_plt_2);
    ax2.set_title("Mouvement brownien pour log(beta_t)");
    ax2.set_xlabel("Temps (en semaines)");
    ax2.set_ylabel("log-contact rate");
    ax2.legend()

    # Partie stochastique, en échelle normale
    ax3.plot(t, np.exp(result[:,4]), c = "blue", label = "Taux de contact", **kwargs_plt_3)
    ax3.set_title("Taux de contact (échelle linéaire)")
    ax3.set_xlabel("Temps (en semaines)")
    ax3.set_ylabel("contact rate")
    ax3.legend()

    # Agrandissement de la fenêtre
    plt.rcParams["figure.figsize"] = size

    plt.show()
    return None



# Discretisation des données
def generate_noisy_prevalence(I_t, sigma_y, bin_size, t):
    """
    Pour un vecteur donné correspondant au nombre de
    personnes infectées, génère des observations
    bruitées tirées selon une Gaussienne

    Les observations sont faites en temps discret
    donc les observations sont discrétisées avec
    un pas semaine, et on prend une des valeurs
    tirées au hasard parmi l'ensemble des valeurs
    tombant dans la boite en question.

    Arguments:
    - solution : un vecteur de réalisations de I_t
    - sigma_y : le bruit (entré comme std dev)
    - bin_size : longueur de la boite (en semaines)
    - t : le vecteur de temps passé en input
    """

    # Déduire le nombre de boites à partir de la longueur
    # du vecteur d'entrée et de la taille des boites souhaitée
    nb_days = int(np.max(t))
    nb_bins = math.floor(nb_days/bin_size) + 1

    # Initialisation du vecteur d'observations bruitées
    # Sous forme de liste, car si on n'a une boite vide,
    # on ne voudra pas l'utiliser. Permet de couvrir tout l'intervalle
    # avec éventuellement une taille de boite "non standard".
    noisy_values = []

    # Découpage du vecteur continu
    # Vecteur des longueurs cumulées, pour "avancer" dans les indices de I_t.
    length_sequences = 0
    indices = []
    for i in range(nb_bins):

        # Définition la condition
        condition = [i * bin_size <= s < (i+1) * bin_size for s in t]
        # Extraire le sous vecteur qui satisfait la condition
        current_bin = np.extract(condition, t)

        # Dans chacune des boites, prendre une valeur de temps au hasard
        # Seulement pour les boites qui ont plus de une observation
        # sinon considérées comme vides
        if len(current_bin) > 0:

            # On sélectionn un indice au hasard
            index_pick = random.randrange(length_sequences, len(current_bin) + length_sequences)
            indices.append(index_pick)
            length_sequences += len(current_bin)
            # Aller chercher cette valeur de vecteur I_t
            baseline_mean = I_t[index_pick]

            # Bruiter l'observation
            y_noisy = float(norm.rvs(loc=baseline_mean, scale=sigma_y, size=1, random_state=None))
            # On ne peut pas avoir de valeurs négatives pour I_t.
            y_noisy = max(0,y_noisy)

            # Ajouter au vecteurs d'observations
            noisy_values.append(y_noisy)
        else:
            continue

    return noisy_values, indices
