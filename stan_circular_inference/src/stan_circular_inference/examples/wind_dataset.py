import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stan_circular_inference.factories.vonmises_factory import VonMisesMK, VonMisesM
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

import numpy as np
import pandas as pd
import time

def run():

    df = pd.read_csv('datasets/wind_dataset.csv')

    mu = Uniform(0, 2*np.pi)
    kappa = Exponential(0.1)

    model1 = VonMisesMK(mu, kappa)
    model2 = VonMisesM(mu)

    values = np.array(df['wind_dir'], dtype='float')

    test_data = {
            "N": 5,
            "values": [1.0, 2.0, 3.0, 4.0, 5.0]
        }

    model1_data = {'N': len(values), 'values': values}
    model2_data = {'N': len(values), 'values': values, 'kappa': kappa}

    service = BayesianInferenceService(model1)

    start_time = time.time()

    interest_parameter_values = service.get_pystan_statistics(data=model1_data, parameters=['mu', 'kappa'], sample_amount=10000)

    #interest_parameter_values = service.get_pystan_statistics(data=model1_data, parameters=['mu', 'kappa'], sample_amount=1000)

    print("--- %s seconds ---" % (time.time() - start_time))

    service.circular_graphic(interest_parameter_values['mu'], min_val=0, max_val=2*np.pi)

if __name__ == "__main__":
    run()