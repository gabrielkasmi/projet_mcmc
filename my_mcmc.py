'''
Subclass of PMMH from particles.mcmc.PMMH to allow storing history of Xt
and eventually, run full algorithm.
'''

import particles
import numpy as np
from scipy import stats
from particles.mcmc import PMMH, GenericRWHM
from particles.smc_samplers import Bootstrap
from particles import smc_samplers as ssp


class my_PMMH(PMMH):
    """Subclass of PMMH where the SMC history of the last accepted iteration is saved.
    """

    def __init__(self, niter=10, seed=None, verbose=0, ssm_cls=None, smc_cls=particles.SMC, prior=None, data=None,
                 smc_options=None, fk_cls=Bootstrap, Nx=100, theta0=None, adaptive=True, scale=1., rw_cov=None):

        super().__init__(niter, seed, verbose, ssm_cls, smc_cls, prior, data, smc_options, fk_cls, Nx, theta0, adaptive,
                         scale, rw_cov)
        self.smc_options = {'summaries': False, 'store_history': True}
        self.prop_history = None
        self.history = None  # stored history (last accepted iteration's history)


    def compute_post(self):
        self.prop.lpost[0] = self.prior.logpdf(self.prop.theta)
        if np.isfinite(self.prop.lpost[0]):
            ssm_kwargs_dict = ssp.rec_to_dict(self.prop.theta[0])
            pf = self.alg_instance(ssm_kwargs_dict)
            pf.run()
            self.prop_history = pf.hist  # store SMC sampler history
            self.prop.lpost[0] += pf.logLt

    def step(self, n):
        z = stats.norm.rvs(size=self.chain.dim)
        self.prop.arr[0] = self.chain.arr[n - 1] + np.dot(self.L, z)
        self.compute_post()
        lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
        if np.log(stats.uniform.rvs()) < lp_acc:  # accept
            self.chain.copyto_at(n, self.prop, 0)
            self.history = self.prop_history  # store particles history
            self.nacc += 1
        else:  # reject
            self.chain.copyto_at(n, self.chain, n - 1)
        if self.adaptive:
            self.cov_tracker.update(self.chain.arr[n])
            self.L = self.scale * self.cov_tracker.L
