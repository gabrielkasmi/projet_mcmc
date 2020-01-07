'''
Subclass of SMC in particles.core to allow passing down additional arguments to PY in the SEIR class
'''

from particles.core import SMC


class my_SMC(SMC):
    """ Copy of the SMC class where
    """

    def __init__(self, fk=None, N=100, seed=None, qmc=False, resampling="systematic", ESSrmin=0.5, store_history=False,
                 verbose=False, summaries=True, **sum_options):

        super().__init__(fk, N, seed, qmc, resampling, ESSrmin, store_history, verbose, summaries, **sum_options)

    def reweight_particles(self):
        """
        Pass down entire X history and number of particles instead of x and xp
        :return: None
        """
        self.wgts = self.wgts.add(self.fk.logG(self.t, self.N, self.hist.X))



