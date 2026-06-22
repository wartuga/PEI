import numpy as np
import pandas as pd
from stan_circular_inference.service.bayesian_inference import BayesianInferenceService

df = pd.read_csv('datasets/wind_dataset.csv')
df = df[(df['wind_dir'] != -9999) | (df['wind_speed'] != -9999)].dropna().copy()

df['wind_dir'] = np.radians(df['wind_dir']) % (2*np.pi)

model = """
    data {
        int<lower=0> N;
        vector[N] direction;
        vector[N] velocity;
    }

    parameters {
        real<lower=0, upper=2*pi()> mu;
        real<lower=0> kappa;
        real<lower=0, upper=1> beta;
    }

    model {
        mu ~ uniform(0, 2*pi());
        kappa ~ gamma(2, 0.5);
        beta ~ normal(0, 5);

        for(n in 1:N){
            real mu_final = mu + beta * velocity[n];
            
            direction[n] ~ von_mises(mu_final, kappa);
        }
    }
"""

service = BayesianInferenceService(model)

model_data = {'N': len(df['wind_dir'].values), 'direction': df['wind_dir'].values, 'velocity': df['wind_speed'].values}

values = service.get_pystan_statistics(data=model_data, parameters=['mu', 'kappa', 'beta'], sample_amount=5000)

service.multiple_graphics(values, min_val=0, max_val=2*np.pi, parameters_type=[True, False, False])