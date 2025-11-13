import nutpie
import asyncio
import numpy as np
import time
import utils

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

start_time = time.time()

count_data = np.loadtxt("txtdata.csv")
count_data = count_data.astype(int).tolist()
n_count_data = len(count_data)
count_sum = sum(count_data)

compiled = (
    nutpie
    .compile_stan_model(code=model)
    .with_data(N=n_count_data, messages_count=count_data)
)

trace = nutpie.sample(compiled_model=compiled, draws=200000, chains=4)

asyncio.run(utils.get_nutpie_statistics(trace, confidence_interval=30))

print("--- %s seconds ---" % (time.time() - start_time))