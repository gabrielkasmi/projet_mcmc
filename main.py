"""
Main script for running the PMMH algorithm in the SEIR context.
"""

import warnings

warnings.filterwarnings('ignore')

# general imports
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd

# imports from subclasses of particles package classes
from my_mcmc import my_PMMH
from my_SMC import my_SMC
from my_state_space_models import *


LABELS = ['S', 'E', 'I', 'R']
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
    res = np.empty(shape=(T * N, 2), dtype=np.float)
    for index, row in enumerate(X):  # fill columns
        res[index * N: (index + 1) * N, 0] = index * np.ones(shape=(N), dtype=np.int)
        res[index * N: (index + 1) * N, 1] = row
    res = pd.DataFrame(data=res, columns=['t', 'X'])  # cast to Dataframe
    return res


if __name__ == '__main__':
    '''
    Visualization of simulated data
    '''

    # import data
    data = pd.read_csv("data_with_beta.csv")
    y = np.array(data['incidence'])
    x = np.array(data['true_beta'])

    # plot Y
    plt.style.use('ggplot')
    plt.plot(y)
    plt.xlabel('t in weeks')
    plt.ylabel('Observed incidence')
    plt.title('Evolution of observed incidence across time')
    plt.show()

    # plot true X
    plt.style.use('ggplot')
    plt.plot(x)
    plt.xlabel('t in weeks')
    plt.ylabel('True X = log(beta)')
    plt.title('Evolution of log contact rate across time')
    plt.show()

    # plot true SEIR graphs
    solver = SEIR_ODE(x)
    sol = solver.solve()
    for i in range(4):
        plt.plot(sol[:, i], label=LABELS[i])
    plt.xlabel('t in weeks')
    plt.title('True SEIR evolution')
    plt.legend()
    plt.show()
    '''
    PMMH run
    '''

    prior_dict = {'sigma': dists.Gamma(1, 1),
                  'tau': dists.Gamma(1, 1)}  # prior distributions of parameters sigma and tau
    prior = dists.StructDist(prior_dict)  # priors of parameters sigma and tau

    my_alg = my_PMMH(ssm_cls=SEIR, smc_cls=my_SMC, fk_cls=my_Bootstrap, niter=100, data=y, Nx=100,
                     prior=prior, verbose=100)  # instantiate PMMH algorithm

    my_alg.run()  # run all iterations

    '''
    Visualization and exploration of results
    '''

    hist = my_alg.history.X  # retrieve results
    if len(hist):

        # plot approximated X
        results = flatten(hist)  # reformat results
        sb.lineplot(x='t', y='X', data=results)  # plot results
        plt.xlabel('t in weeks')
        plt.ylabel('Approximated X = log(beta)')
        plt.show()

        # plot approximated SEIR
        full_seir = []
        for part in np.array(hist).T:
            solver = SEIR_ODE(part)
            sol = solver.solve()
            full_seir.append(sol)
        full_seir = np.transpose(np.array(full_seir), axes=(2, 1, 0))
        for i, res in enumerate(full_seir):
            ax = sb.lineplot(x='t', y='X', data=flatten(res), label=LABELS[i])
            plt.plot()
        plt.show()


    else:
        print('No proposition of parameters was accepted, impossible to plot log contact rate')

