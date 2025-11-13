import stan
import asyncio
import numpy as np
import utils

test_model = """
data {
  int<lower=0> N;
  array[N] int<lower=0> y;
}
parameters {
  real mu;
}
model {
  y ~ poisson(mu);
}
"""

test_data = {"N": 3, "y": [1, 2, 3]}

# Build the model
posterior = stan.build(test_model, data=test_data)

#print(type(posterior).__module__)

# Sample from the posterior model
fit = posterior.sample(num_chains=4, num_samples=10000)

asyncio.run(utils.get_pystan_statistics(fit, test_model))