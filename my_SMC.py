'''
Subclass of SMC in particles.core to allow passing down additional arguments to PY in the SEIR class
'''

from particles.core import SMC
import numpy as np
import numpy.random as random
import scipy

from particles import utils
from particles import resampling as rs
from particles import collectors
from particles import smoothing
from particles import qmc
from particles import hilbert


class my_SMC(SMC):
    """ particle filter class, with different methods to resample, reweight, etc
        the particle system.

        Parameters
        ----------
        fk: FeynmanKac object
            Feynman-Kac model which defines which distributions are
            approximated
        N: int, optional (default=100)
            number of particles 
        seed: int, optional 
            used to seed the random generator when starting the algorithm
            default=None, in that case the rng is not seeded
        qmc: {True, False}
            if True use the Sequential quasi-Monte Carlo version (the two
            options resampling and ESSrmin are then ignored)
        resampling: {'multinomial', 'residual', 'stratified', 'systematic', 'ssp'}
            the resampling scheme to be used (see `resampling` module for more
            information; default is 'systematic')
        ESSrmin: float in interval [0, 1], optional
            resampling is triggered whenever ESS / N < ESSrmin (default=0.5)
        store_history: bool (default = False)
            whether to store the complete history; see module `smoothing`
        verbose: bool, optional
            whether to print basic info at every iteration (default=False)
        summaries: bool, optional (default=True)
            whether summaries should be collected at every time. 
        **summaries_opts: dict
            options that determine which summaries collected at each iteration 
            (e.g. moments, on-line smoothing estimates); see module ``collectors``

        Attributes
        ----------

        t : int
           current time step
        X : typically a (N,) or (N, d) ndarray (but see documentation) 
            the N particles 
        A : (N,) ndarray (int)
           ancestor indices: A[n] = m means ancestor of X[n] has index m
        wgts: Weigts object 
            An object with attributes lw (log-weights), W (normalised weights)
            and ESS (the ESS of this set of weights) that represents 
            the main (inferential) weights
        aux: Weights object 
            the auxiliary weights (for an auxiliary PF, see FeynmanKac)
        cpu_time : float
            CPU time of complete run (in seconds)
        hist: `ParticleHistory` object (None if option history is set to False)
            complete history of the particle system; see module `smoothing`
        summaries: `Summaries` object (None if option summaries is set to False)
            each summary is a list of estimates recorded at each iteration. The
            following summaries are computed by default: 
                + ESSs (the ESS at each time t)
                + rs_flags (whether resampling was performed or not at each t)
                + logLts (estimates of the normalising constants)
            Extra summaries may also be computed (such as moments and online 
            smoothing estimates), see module `collectors`. 

        Methods
        -------
        run():
            run the algorithm until completion
        step()
            run the algorithm for one step (object self is an iterator)

    """

    def __init__(self,
                 fk=None,
                 N=100,
                 seed=None,
                 qmc=False,
                 resampling="systematic",
                 ESSrmin=0.5,
                 store_history=False,
                 verbose=False,
                 summaries=True,
                 **sum_options):

        self.fk = fk
        self.N = N
        self.seed = seed
        self.qmc = qmc
        self.resampling = resampling
        self.ESSrmin = ESSrmin
        self.verbose = verbose

        # initialisation
        self.t = 0
        self.rs_flag = False  # no resampling at time 0, by construction
        self.logLt = 0.
        self.wgts = rs.Weights()
        self.aux = None
        self.X, self.Xp, self.A = None, None, None

        # summaries computed at every t
        if summaries:
            self.summaries = collectors.Summaries(**sum_options)
        else:
            self.summaries = None
        if store_history:
            self.hist = smoothing.ParticleHistory(self.fk, self.N)
        else:
            self.hist = None

    def __str__(self):
        return self.fk.summary_format(self)

    @property
    def W(self):
        return self.wgts.W

    def reset_weights(self):
        """Reset weights after a resampling step.
        """
        if self.fk.isAPF:
            lw = (rs.log_mean_exp(self.logetat, W=self.W)
                  - self.logetat[self.A])
            self.wgts = rs.Weights(lw=lw)
        else:
            self.wgts = rs.Weights()

    def setup_auxiliary_weights(self):
        """Compute auxiliary weights (for APF).
        """
        if self.fk.isAPF:
            self.logetat = self.fk.logeta(self.t - 1, self.X)
            self.aux = self.wgts.add(self.logetat)
        else:
            self.aux = self.wgts

    def generate_particles(self):
        if self.qmc:
            # must be (N,) if du=1
            u = qmc.sobol(self.N, self.fk.du).squeeze()
            self.X = self.fk.Gamma0(u)
        else:
            self.X = self.fk.M0(self.N)

    def reweight_particles(self):
        self.wgts = self.wgts.add(self.fk.logG(self.t, self.N, self.hist.X))

    def resample_move(self):
        self.rs_flag = self.aux.ESS < self.N * self.ESSrmin
        if self.rs_flag:  # if resampling
            self.A = rs.resampling(self.resampling, self.aux.W)
            self.Xp = self.X[self.A]
            self.reset_weights()
            self.X = self.fk.M(self.t, self.Xp)
        elif not self.fk.mutate_only_after_resampling:
            self.A = np.arange(self.N)
            self.Xp = self.X
            self.X = self.fk.M(self.t, self.Xp)

    def resample_move_qmc(self):
        self.rs_flag = True  # we *always* resample in SQMC
        u = qmc.sobol(self.N, self.fk.du + 1)
        tau = np.argsort(u[:, 0])
        h_order = hilbert.hilbert_sort(self.X)
        if self.hist is not None:
            self.hist.h_orders.append(h_order)
        self.A = h_order[rs.inverse_cdf(u[tau, 0], self.aux.W[h_order])]
        self.Xp = self.X[self.A]
        v = u[tau, 1:].squeeze()
        #  v is (N,) if du=1, (N,d) otherwise
        self.X = self.fk.Gamma(self.t, self.Xp, v)
        self.reset_weights()

    def compute_summaries(self):
        if self.t > 0:
            prec_log_mean_w = self.log_mean_w
        self.log_mean_w = rs.log_mean_exp(self.wgts.lw)
        if self.t == 0 or self.rs_flag:
            self.loglt = self.log_mean_w
        else:
            self.loglt = self.log_mean_w - prec_log_mean_w
        self.logLt += self.loglt
        if self.verbose:
            print(self)
        if self.summaries:
            self.summaries.collect(self)
        if self.hist:
            self.hist.save(X=self.X, w=self.wgts, A=self.A)

    def __next__(self):
        """One step of a particle filter.
        """
        if self.fk.done(self):
            raise StopIteration
        if self.t == 0:
            if self.seed:
                random.seed(self.seed)
            self.generate_particles()
        else:
            self.setup_auxiliary_weights()  # APF
            if self.qmc:
                self.resample_move_qmc()
            else:
                self.resample_move()
        self.reweight_particles()
        self.compute_summaries()
        self.t += 1

    def next(self):
        return self.__next__()  #  Python 2 compatibility

    def __iter__(self):
        return self

    @utils.timer
    def run(self):
        """Runs particle filter until completion.

           Note: this class implements the iterator protocol. This makes it
           possible to run the algorithm step by step::

               pf = SMC(fk=...)
               next(pf)  # performs one step
               next(pf)  # performs one step
               for _ in range(10):
                   next(pf)  # performs 10 steps
               pf.run()  # runs the remaining steps

            In that case, attribute `cpu_time` records the CPU cost of the last
            command only.
        """
        for _ in self:
            pass


