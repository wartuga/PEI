import stan_circular_inference.utils as utils
import numpy as np
from scipy.stats import vonmises

model = """
data {
    int<lower=0> N;
    vector[N] values;
    // Mixing weight for the first component (between 0 and 1)
    real<lower=0, upper=1> mixing_weight;
}

parameters {
    real<lower=0, upper=2*pi()> mu1;
    real<lower=-pi()/2, upper=0> mu2;
    
    real<lower=0> kappa1;
    real<lower=0> kappa2;
    
    // Optional: If you want to estimate the mixing weight and remove it from data
    // real<lower=0, upper=1> mixing_weight;
}

transformed parameters {
    // real<lower=0, upper=2*pi()> mu;

    // Compute circular mean
    // mu = atan2(sin(mu1) + sin(mu2), cos(mu1) + cos(mu2));
    
    // Adjust to [0, 2π) if needed
    // if (mu < 0) {
    //     mu = mu + 2 * pi();
    // }
}

model {
    // Priors for the parameters
    mu1 ~ normal(0, pi()/2);
    mu2 ~ uniform(-pi()/2, 0);
    kappa1 ~ exponential(0.1);
    kappa2 ~ exponential(0.1);
    
    // Mixture model likelihood using log_sum_exp (it prevents underflow and overflow)
    for (n in 1:N) {
        target += log_sum_exp(
            log(mixing_weight) + von_mises_lpdf(values[n] | mu1, kappa1),
            log(1 - mixing_weight) + von_mises_lpdf(values[n] | mu2, kappa2)
        );
    }
}
"""

mu1 = np.pi/2
mu2 = -np.pi/2
kappa = 5
size = 100

samples1 = vonmises.rvs(kappa, loc=mu1, size=size)
samples2 = vonmises.rvs(kappa, loc=mu2, size=size)

samples = np.concatenate([samples1, samples2])
np.random.shuffle(samples)

data = {'N': len(samples), 'values': samples, 'mixing_weight': 0.5}
values = utils.get_pystan_statistics(model=model, model_data=data, parameters=['mu1', 'mu2'], sample_amount=1000)
utils.circular_graphic(values['mu1'], min_val=0, max_val=2*np.pi)
utils.circular_graphic(values['mu2'], min_val=0, max_val=2*np.pi)