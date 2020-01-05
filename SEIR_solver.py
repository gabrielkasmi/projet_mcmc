"""
TO-DO:
- corriger les méthodes pour bien calculer la prévalence et pouvoir ainsi la réutiliser pour P(Y|X)
- tester sur data_with_beta
"""

import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt

class SEIR_ODE:

    def __init__(self, X, coefs=[0.2, 0.01], init=[0.7, 0., 0.3, 0.]):
        self.coefs = coefs  # coefs k and gamma in the article
        self.init = init  # inital SEIR conditions
        self.X = X  # X trajectories (log_beta)

    def beta(self, t):
        res = np.exp(np.interp(np.array([t]), np.arange(len(self.X)), self.X))[0]
        return res

    def prevalence_solve(self):
        t_grid = np.arange(max(len(self.X) - 1, 1))
        solution = np.array(odeint(self.simulate_SEIR_model, self.init, t_grid,
                                                   args=(self.beta,)))  # solve full ODE
        return solution,t_grid

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

data = pd.read_csv("data_with_beta.csv")
y=np.array(data['incidence'])
x=np.array(data['true_beta'])

new_solver = SEIR_ODE(x)
solution, grid = new_solver.prevalence_solve()
plt.plot(solution[:, 0], label='S')
plt.plot(solution[:, 1], label='E')
plt.plot(solution[:, 2], label='I')
plt.plot(solution[:, 3], label='R')
plt.legend()
plt.show()