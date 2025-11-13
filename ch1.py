import stan
import numpy as np
import utils
import time

model = """
data { 
  int<lower=0> N;
  array[N] int<lower=0> messages_count;
} 
parameters {
  real<lower=1> switch;
  real<lower=0> lambda1;
  real<lower=0> lambda2;
}
transformed parameters {
  array[N] real<lower=0> lambda;
  for (n in 1:N) {
    if (n <= switch)
      lambda[n] = lambda1;
    else
      lambda[n] = lambda2;
  }
}
model {
  switch ~ uniform(0, N);
  lambda1 ~ exponential(0.01);
  lambda2 ~ exponential(0.02);
  messages_count ~ poisson(lambda);
}
"""

# 0.01 & 0.02

start_time = time.time()

count_data = np.loadtxt("txtdata.csv")
alpha = 1.0/count_data.mean() # 0.05
count_data = count_data.astype(int).tolist()
n_count_data = len(count_data)
data = {'N': n_count_data, 'messages_count': count_data}

# Build the model
posterior = stan.build(model, data=data)

# Sample from the posterior model
fit = posterior.sample(num_chains=4, num_samples=70000)

utils.get_pystan_statistics(fit=fit)

print("--- %s seconds ---" % (time.time() - start_time))