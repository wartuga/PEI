import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stan_circular_inference.factories.vonmises_factory import VonMises_MK, VonMises_M
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.utils import utils
import numpy as np
import pandas as pd
import time

def run():

    df = pd.read_csv('datasets/wind_dataset.csv')

    mu = Uniform(0, 2*np.pi)
    kappa = Exponential(0.1)

    model1 = VonMises_MK(mu, kappa)
    model2 = VonMises_M(mu)

    values = np.array(df['wind_dir'], dtype='float')

    model1_data = {'N': len(values), 'values': values}
    model2_data = {'N': len(values), 'values': values, 'kappa': kappa}

    start_time = time.time()

    interest_parameter_values = utils.get_pystan_statistics(model_data=model1_data, model=model1.gen_stan_model(), parameters=['mu', 'kappa'], sample_amount=1000)

    print("--- %s seconds ---" % (time.time() - start_time))

    utils.circular_graphic(interest_parameter_values['mu'], min_val=0, max_val=2*np.pi)

if __name__ == "__main__":
    run()