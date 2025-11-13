import stan
import numpy as np
import utils
import time
import matplotlib.pyplot as plt

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
fit = posterior.sample(num_chains=4, num_samples=1000)

stats = utils.get_pystan_statistics(fit=fit)

print("--- %s seconds ---" % (time.time() - start_time))

switch_stats = stats.loc["switch", ["mean", "sd", "r_hat"]]

# Get statistic parameters of interest
mean = switch_stats['mean']
sd = switch_stats['sd']
r_hat = switch_stats['r_hat']

# Get the distribution data
possible_values, distribution = utils.distribute_probability(n_count_data, mean, sd, r_hat)

# Create even angles
angles = np.linspace(0, 2 * np.pi, n_count_data, endpoint=False)

# Create a figure with polar projection
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Draw and fill the area of interest
ax.fill(angles, distribution, alpha=0.3, color='tomato')
ax.plot(angles, distribution, color='tomato', linewidth=2)

# Add points for each angle
ax.scatter(angles, distribution, color='tomato', s=30, zorder=3)

# Add points around the circle
for i, (angle, value, possible_value) in enumerate(zip(angles, distribution, possible_values)):
    label_radius = max(distribution) * 1.1  # Point label radius
    ax.text(angle, label_radius, f"{possible_value}", 
            ha='center', va='center', fontsize=8)

# Graphic configurations
ax.set_theta_offset(np.pi/2)  # Start at the top
ax.set_theta_direction(-1)    # Clockwise
ax.set_ylim(0, max(distribution) * 1.2)
ax.grid(True)
plt.title('Diagrama de Rosas')
plt.show()