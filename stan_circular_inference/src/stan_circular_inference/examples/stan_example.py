import stan
import asyncio
import arviz as az
import pandas as pd
import nest_asyncio
import stan_circular_inference.utils.utils as utils

# Needed to run the code using a WSL terminal or WSL extension in VSCode
# It allows nested event loops to run
nest_asyncio.apply()

model = """
data { 
  int<lower=1> n1; 
  int<lower=1> n2; 
  int<lower=0> k1;
  int<lower=0> k2;
} 
parameters {
  real<lower=0,upper=1> theta1;
  real<lower=0,upper=1> theta2;
} 
transformed parameters {
  real<lower=-1,upper=1> delta;
  delta = theta1 - theta2;
}
model {
  theta1 ~ beta(1, 1);
  theta2 ~ beta(1, 1);
  k1 ~ binomial(n1, theta1);
  k2 ~ binomial(n2, theta2);
}
"""

data = {'k1':5, 'n1':10, 'k2':7, 'n2':10}

# Run and await the main async function
asyncio.run(utils.get_pystan_statistics(data, model))