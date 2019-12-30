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
import scipy

# imports from the package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists
from my_mcmc import my_PMMH
from my_SMC import my_SMC
from my_state_space_models import *

warnings.filterwarnings('ignore')


def flatten(X):
    """
    Process list of list X to pd.DataFrame to easily visualize results
    :param X: list of list
    :return: pd.DataFrame with columns
        t : time index to aggregate on
        X : particle X value at time t
    """
    T = len(X)
    N = np.shape(X[0])[0]
    res = np.empty(shape=(T*N, 2), dtype=np.float)
    for index, row in enumerate(X):  # fill columns
        res[index * N: (index + 1) * N, 0] = index * np.ones(shape=(N), dtype=np.int)
        res[index * N: (index + 1) * N, 1] = row
    res = pd.DataFrame(data=res, columns=['t', 'X'])  # cast to Dataframe
    return res


'''
Creation and simulation of state space model 
'''
# my_ssm = SEIR()  # use default values for all parameters
# x, y = my_ssm.simulate(100)  # simulate Xt and Yt

data = pd.read_csv("generative_prevalence.csv")
y=np.array(data['incidence'])

plt.style.use('ggplot')
plt.plot(y)
plt.xlabel('t')
plt.ylabel('data')
plt.show()

'''
PMMH run
'''

prior_dict = {'sigma': dists.Gamma(1, 1), 'tau': dists.Gamma(1, 1)}
prior = dists.StructDist(prior_dict)  # priors of parameters sigma and tau

my_alg = my_PMMH(ssm_cls=SEIR_hard, smc_cls=my_SMC, fk_cls=my_Bootstrap, niter=30, data=y, Nx=15, prior=prior)  # instantiate PMMH algorithm

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