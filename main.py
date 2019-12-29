'''
TO-DO:
-   give access to either all history of Xt (for solving ODE from t = 0)
    or only Vt-1 (to have initial conditions for one step of ODE) to function incidence
-   fill function incidence according to the choice of previous arguments
-   optional: store full solution of ODE for plot SEIR evolution with error bands
'''

import warnings

# the usual imports
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

# imports from the package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from my_mcmc import my_PMMH

warnings.filterwarnings('ignore')


class SEIR(ssm.StateSpaceModel):

    default_params = {'sigma': 1, 'tau': 1}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def PX(self, t, xp):  # Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=xp, scale=self.sigma)

    def incidence(self, x):
        # WARNING: not filled yet, set to random function as example
        return x

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Normal(loc=self.incidence(x), scale=self.tau)


def flatten(X):
    '''
    Process list of list X to pd.DataFrame to easily visualize results
    :param X: list of list
    :return: pd.DataFrame with columns
        t : time index to aggregate on
        X : particle X value at time t
    '''
    T = np.shape(X)[0]
    N = np.shape(X)[1]
    res = np.empty(shape=(T*N, 2), dtype=np.float)
    for index, row in enumerate(X):  # fill columns
        res[index * N: (index + 1) * N, 0] = index * np.ones(shape=(100), dtype=np.int)
        res[index * N: (index + 1) * N, 1] = np.array(row)
    res = pd.DataFrame(data=res, columns=['t', 'X'])  # cast to Dataframe
    return res


'''
Creation and simulation of state space model 
'''
my_ssm = SEIR()  # use default values for all parameters
x, y = my_ssm.simulate(100)  # simulate Xt and Yt

plt.style.use('ggplot')
plt.plot(x)
plt.xlabel('t')
plt.ylabel('data')
plt.show()

'''
PMMH run
'''

prior_dict = {'sigma': dists.Gamma(1, 1), 'tau': dists.Gamma(1, 1)}
prior = dists.StructDist(prior_dict)  # priors of parameters sigma and tau

my_alg = my_PMMH(ssm_cls=SEIR, niter=100, data=y, Nx=100, prior=prior)  # instantiate PMMH algorithm

my_alg.run()  # run all iterations


'''
Visualization and exploration of results
'''

hist = my_alg.history.X
results = flatten(hist)  # reformat results
sb.lineplot(x='t', y='X', data=results)  # plot results
plt.show()

'''
Both plots should look the same
'''