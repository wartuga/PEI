from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.distributions_factory import Uniform
from stan_circular_inference.service import BayesianInferenceService

import numpy as np
import pandas as pd
import time

df = pd.read_csv('src/datasets/wind_dataset.csv')

mu = Uniform(0, 2*np.pi)
rho = Uniform(-0.1, 0.1)

model1 = Cardioid(mu, rho)

values = np.array(df['wind_dir'], dtype='float')

test_values = [-np.pi/4, np.pi/4]

test_data = {
        "N": 2,
        "values": [-np.pi/4, np.pi/4]
    }

model1_data = {'N': len(values), 'values': values}

service = BayesianInferenceService(model1)

start_time = time.time()

posterior = service.build_model(test_data)

interest_parameter_values = service.get_pystan_statistics(data=test_data, parameters=['mu', 'rho'], sample_amount=5000)

print("--- %s seconds ---" % (time.time() - start_time))

service.match_points_to_distributions(test_values, test_values, interest_parameter_values, min_val=0, max_val=2*np.pi)

service.circular_graphic(interest_parameter_values['mu'], min_val=0, max_val=2*np.pi)

service.multiple_graphics(interest_parameter_values, min_val=0, max_val=2*np.pi, param_names=['mu', 'rho'], parameters_type=[True, True])