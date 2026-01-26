import numpy as np
from scipy.stats import vonmises
import time
import stan_circular_inference.utils.utils as utils

model = """
data {
  int<lower=1> N;
  array[N] real values;
  real<lower=0> kappa;
}
parameters {
  real<lower=0, upper=2*pi()> mu;
}
model {
  real normalizer = log(2 * pi() * modified_bessel_first_kind(0, kappa));
  
  for (n in 1:N) {
    target += kappa * cos(values[n] - mu) - normalizer;
  }
  
  mu ~ uniform(0, 2*pi());
}
"""

mu_real = np.pi/4
kappa_real = 15
data = vonmises.rvs(kappa_real, loc=mu_real, size=100)

model_data = {'N': len(data), 'values': data, 'kappa': kappa_real}

init = [{'mu': float(mu_real)} for _ in range(4)]

start_time = time.time()
interest_parameter_values = utils.get_pystan_statistics(model_data=model_data, model=model, parameter='mu', sample_amount=100000, init=init)

print(f"Time taken: {time.time() - start_time:.2f} seconds")

utils.circular_graphic(interest_parameter_values, data, min_val=0.7, max_val=0.9, density=True)