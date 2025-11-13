import pymc as pm
import numpy as np
import asyncio
import utils
import time

timer = time.time()

count_data = np.loadtxt("txtdata.csv")
n_count_data = len(count_data)

with pm.Model() as model:
    alpha = 1.0/count_data.mean()  # Recall count_data is the
                                   # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)


with model:
    idx = np.arange(n_count_data) # Index
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    observation = pm.Poisson("obs", lambda_, observed=count_data)

with model:
    step = pm.Metropolis()
    # 100000 && 5000
    trace = pm.sample(100000, tune=5000, step=step, return_inferencedata=True)

asyncio.run(utils.get_nutpie_statistics(trace))

print("%ss" % (time.time() - timer))