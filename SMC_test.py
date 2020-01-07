import numpy as np
import pandas as pd

from my_SMC import my_SMC
from my_state_space_models import my_Bootstrap, SEIR

import warnings

warnings.filterwarnings('ignore')


data = pd.read_csv("data_with_beta.csv")
y = np.array(data['incidence'])

smc = my_SMC(fk=my_Bootstrap(ssm=SEIR(sigma=0.07, tau=0.2), data=y), N=100, store_history='True')

smc.run()