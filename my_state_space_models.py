'''
Personal state space models
'''

# the usual imports
import numpy as np
import pandas as pd
import scipy

# imports from the package
from particles import state_space_models as ssm
from particles import distributions as dists

class SEIR_ODE:

    def __init__(self, X, coefs=[0.3, 0.2], init=[0.9, 0., 0.1, 0.]):
        self.coefs = coefs  # coefs k and gamma in the article
        self.init = init  # inital SEIR conditions
        self.X = X  # X trajectories

    def beta(self, t):
        beta = np.exp[self.X]
        return

    def prevalence_solve(self, t):
        t_grid = np.linspace(0, t, 1)
        solution = np.array(scipy.integrate.odeint(self.simulate_SEIR_model, self.init, t_grid,
                                          args=(self.beta, *self.coefs)))  # solve full ODE
        prevalence = solution[-1, 2]  # get last E
        return prevalence

    def simulate_SEIR_model(y, t, beta, gamma, sigma):
        S, E, I, R = y
        N = S + E + I + R

        # Equations
        dS_dt = - beta(t) * S * I / N
        dE_dt = beta(t) * S * I / N - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I

        return ([dS_dt, dE_dt, dI_dt, dR_dt])


class SEIR_hard(ssm.StateSpaceModel):

    default_params = {'sigma': 1, 'tau': 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=xp, scale=self.sigma)

    def prevalence(self, t, N, trajectory):
        prev_vect = np.ones((N,))
        if trajectory == []:
            prev_vect *= SEIR_ODE(None).init[2]
        else:
            for particle_index in range(N):
                ODE = SEIR_ODE([particles[particle_index] for particles in trajectory])
                solution = ODE.prevalence_solve(t)
                prev_vect[particle_index] = solution
        return prev_vect

    def PY(self, t, N, trajectory):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Normal(loc=self.prevalence(t, N, trajectory), scale=self.tau)


class SEIR(ssm.StateSpaceModel):

    default_params = {'sigma': 1, 'tau': 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=xp, scale=self.sigma)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Normal(loc=x, scale=self.tau)


class my_Bootstrap(ssm.Bootstrap):
    """Bootstrap Feynman-Kac formalism of a given state-space model.

        Parameters
        ----------

        ssm: `StateSpaceModel` object
            the considered state-space model
        data: list-like
            the data

        Returns
        -------
        `FeynmanKac` object
            the Feynman-Kac representation of the bootstrap filter for the
            considered state-space model
        """

    def __init__(self, ssm=None, data=None):
        self.ssm = ssm
        self.data = data
        self.du = self.ssm.PX0().dim

    @property
    def T(self):
        return 0 if self.data is None else len(self.data)

    def M0(self, N):
        return self.ssm.PX0().rvs(size=N)

    def M(self, t, xp):
        return self.ssm.PX(t, xp).rvs(size=xp.shape[0])

    def logG(self, t, N, x_history):
        return self.ssm.PY(t, N, x_history).logpdf(self.data[t])

    def Gamma0(self, u):
        return self.ssm.PX0().ppf(u)

    def Gamma(self, t, xp, u):
        return self.ssm.PX(t, xp).ppf(u)

    def logpt(self, t, xp, x):
        """PDF of X_t|X_{t-1}=xp"""
        return self.ssm.PX(t, xp).logpdf(x)

    def upper_bound_trans(self, t):
        return self.ssm.upper_bound_log_pt(t)

    def add_func(self, t, xp, x):
        return self.ssm.add_func(t, xp, x)