"""
TO-DO:
- corriger les méthodes pour bien calculer la prévalence et pouvoir ainsi la réutiliser pour P(Y|X)
- tester sur data_with_beta
"""

import numpy as np
import scipy

class SEIR_ODE:

    def __init__(self, X, coefs=[0.3, 0.2], init=[0.9, 0., 0.1, 0.]):
        self.coefs = coefs  # coefs k and gamma in the article
        self.init = init  # inital SEIR conditions
        self.X = X  # X trajectories

    def beta(self, t):
        res = np.exp(self.X[int(t)])
        return res

    def prevalence_solve(self, t):
        t_grid = np.arange(max(len(self.X) - 1, 1))
        solution = np.array(scipy.integrate.odeint(self.simulate_SEIR_model, self.init, t_grid,
                                                   args=(self.beta,)))  # solve full ODE
        return solution

    def prevalence(self, solution):
        """Retourne la prévalence au dernier temps t en fonction de la solution à l'ODE
        calculée juste au dessus
        """
        return None

    def simulate_SEIR_model(self, y, t, beta):
        sigma, gamma = self.coefs[0], self.coefs[1]
        S, E, I, R = y
        N = S + E + I + R

        # Equations
        dS_dt = - beta(t) * S * I / N
        dE_dt = beta(t) * S * I / N - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I

        return ([dS_dt, dE_dt, dI_dt, dR_dt])