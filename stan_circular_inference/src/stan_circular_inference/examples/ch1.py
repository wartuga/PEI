import stan_circular_inference.utils.utils as utils
import numpy as np
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

# 0.0135

count_data = np.loadtxt("txtdata.csv")
mean_data = 1.0/count_data.mean() # 0.05
count_data = count_data.astype(int).tolist()
n_count_data = len(count_data)
count_sum = sum(count_data)
data = {'N': n_count_data, 'messages_count': count_data}

start_time = time.time()

interest_parameter_values = utils.get_pystan_statistics(data=data, model=model, sample_amount=20000)

print("--- %s seconds ---" % (time.time() - start_time))

utils.circular_graphic(interest_parameter_values, count_data, 24)