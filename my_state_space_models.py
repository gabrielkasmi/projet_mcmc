'''
Personal state space models
'''

# the usual imports
import numpy as np

# imports from the package
from particles import state_space_models as ssm
from particles import distributions as dists
from scipy.integrate import odeint


class SEIR_ODE:

    def __init__(self, X, coefs=[0.1, 0.1], init=[0.7, 0., 0.3, 0.]):
        self.coefs = coefs  # coefs k and gamma in the article
        self.init = init  # inital SEIR conditions
        self.X = X  # X trajectories (log_beta)

    def beta(self, t):
        """
        Gives interpolated beta at time t given particles X
        :param t: 
        :return: 
        """
        res = np.interp(np.array([t]), np.arange(len(self.X)), np.exp(self.X))[0]
        return res

    def solve(self):
        """
        Solves the SEIR ODE knowing X
        :return: S, E, I, R values on the same time grid as X in array of shape (T, 4)
        """
        t_grid = np.arange(len(self.X))
        solution = np.array(odeint(self.simulate_SEIR_model, self.init, t_grid, hmin=1E-7,
                                   args=(self.beta,)))  # solve full ODE
        return solution

    def incidence(self, solution):
        """
        Gives last incidence of an array describing SEIR evolution
        :param solution: SEIR evolution (array of shape (T, 4))
        :return: incidence as scalar value
        """
        return solution[-1, 1]

    def simulate_SEIR_model(self, y, t, beta):
        """
        Private function to describe SEIR model
        """
        # Initial conditions
        sigma, gamma = self.coefs[0], self.coefs[1]
        S, E, I, R = y
        N = S + E + I + R

        # Equations
        dS_dt = - beta(t) * S * I / N
        dE_dt = beta(t) * S * I / N - sigma * E
        dI_dt = sigma * E - gamma * I
        dR_dt = gamma * I

        return ([dS_dt, dE_dt, dI_dt, dR_dt])


class SEIR(ssm.StateSpaceModel):
    default_params = {'sigma': 1, 'tau': 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=xp, scale=self.sigma)

    def incidence(self, N, trajectory):
        inc_vect = np.ones((N,))  # initialize vector of one of correct shape
        if trajectory == []:
            inc_vect *= SEIR_ODE(None).init[1]  # return initial incidence for all particles
        else:
            for particle_index in range(N):
                ODE = SEIR_ODE([particles[particle_index] for particles in trajectory])
                solution = ODE.incidence(ODE.solve())  # save incidence for particle
                inc_vect[particle_index] = solution  # fill in inc_vect with incidence value
        return inc_vect

    def PY(self, N, trajectory):  # Distribution of Y_t given full history of X
        return dists.Normal(loc=self.incidence(N, trajectory), scale=self.tau)


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
        super().__init__(ssm, data)

    def logG(self, t, N, x_history):
        return self.ssm.PY(N, x_history).logpdf(self.data[t])
