'''
Subclass of PMMH from particles.mcmc.PMMH to allow storing history of Xt
and eventually, run full algorithm.
'''

import particles
import numpy as np
from particles.mcmc import PMMH, GenericRWHM
from particles.smc_samplers import Bootstrap
from particles import smc_samplers as ssp
from scipy import stats


class my_PMMH(PMMH):
    """Particle Marginal Metropolis Hastings.

    PMMH is class of Metropolis samplers where the intractable likelihood of
    the considered state-space model is replaced by an estimate obtained from
    a particle filter.
    """

    def __init__(self, niter=10, seed=None, verbose=0, ssm_cls=None,
                 smc_cls=particles.SMC, prior=None, data=None, smc_options=None,
                 fk_cls=Bootstrap, Nx=100, theta0=None, adaptive=True, scale=1.,
                 rw_cov=None):
        """
        Parameters
        ----------
        niter: int
            number of iterations
        seed: int (default=None)
            PRNG seed (if None, PRNG is not seeded)
        verbose: int (default=0)
            print some info every `verbose` iterations (never if 0)
        ssm_cls: StateSpaceModel class
            the considered parametric class of state-space models
        smc_cls: class (default: particles.SMC)
            SMC class
        prior: StructDist
            the prior
        data: list-like
            the data
        smc_options: dict
            options to pass to class SMC
        fk_cls: (default=Bootstrap)
            FeynmanKac class associated to the model
        Nx: int
            number of particles (for the particle filter that evaluates the
            likelihood)
        theta0: structured array of length=1
            starting point (generated from prior if =None)
        adaptive: bool
            whether to use the adaptive version
        scale: positive scalar (default = 1.)
            in the adaptive case, covariance of the proposal is scale^2 times
            (2.38 / d) times the current estimate of the target covariance
        rw_cov: (d, d) array
            covariance matrix of the random walk proposal (set to I_d if None)
        """
        self.ssm_cls = ssm_cls
        self.smc_cls = smc_cls
        self.fk_cls = fk_cls
        self.prior = prior
        self.data = data
        # do not collect summaries, no need
        self.smc_options = {'summaries': False, 'store_history': True}
        if smc_options is not None:
            self.smc_options.update(smc_options)
        self.Nx = Nx
        GenericRWHM.__init__(self, niter=niter, seed=seed, verbose=verbose,
                             theta0=theta0, adaptive=adaptive, scale=scale,
                             rw_cov=rw_cov)
        self.prop_history = None  # proposition of history, same as current history,
        # same as trajectories of last iteration
        self.history = []  # stored history (last accepted iteration's history)

    def alg_instance(self, theta):
        return self.smc_cls(fk=self.fk_cls(ssm=self.ssm_cls(**theta),
                                           data=self.data),
                            N=self.Nx, **self.smc_options)

    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            ssm_kwargs_dict = ssp.rec_to_dict(self.prop.theta[0])
            pf = self.alg_instance(ssm_kwargs_dict)
            pf.run()
            self.prop_history = pf.hist
            self.prop.lpost[0] += pf.logLt

    def step(self, n):
        z = stats.norm.rvs(size=self.chain.dim)
        self.prop.arr[0] = self.chain.arr[n - 1] + np.dot(self.L, z)
        self.compute_post()
        lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
        if np.log(stats.uniform.rvs()) < lp_acc:  # accept
            self.chain.copyto_at(n, self.prop, 0)
            self.history = self.prop_history  # store particles
            self.nacc += 1
        else:  # reject
            self.chain.copyto_at(n, self.chain, n - 1)
        if self.adaptive:
            self.cov_tracker.update(self.chain.arr[n])
            self.L = self.scale * self.cov_tracker.L
