import stan
import asyncio
import arviz as az
import pandas as pd
import nest_asyncio
import utils

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

async def get_statistics(fit, confidence_interval=30):
  # Build the model
  #posterior = stan.build(model, data=data)

  # Sample from the posterior model
  #fit = posterior.sample(num_chains=4, num_samples=sample_amount)

  # Convert to ArviZ InferenceData object
  az_data = az.from_pystan(fit)

  # First get the default summary (includes mean, sd, ess, r_hat)
  summary_df = az.summary(az_data, round_to=2)

  # Initialize the minimum value for the confidence interval
  min_val = None
  # Initialize manual percentiles variable
  percentiles = None

  # If the confidence interval is different from the default 3%
  if confidence_interval != 3:

    min_val = min(confidence_interval, 100 - confidence_interval)

    # Add the confidence intervals manually
    percentiles = pd.DataFrame({param: utils.get_percentiles(fit[param], min_val) for param in fit.keys()}).T

  if percentiles is not None:
    # Combine with ArviZ summary
    summary_df = pd.concat([summary_df, percentiles], axis=1)

    # Drop the default columns if different from the desired ones
    summary_df.drop(columns=["hdi_3%", "hdi_97%"], inplace=True)

    print(
      summary_df[["mean", "sd", f"hdi_{min_val}%", f"hdi_{100 - min_val}%", "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]]
      if confidence_interval != 50 
      else summary_df[["mean", "sd", "hdi_50%", "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]]
    )
  
  else:
    print(summary_df)

# Run and await the main async function
#asyncio.run(main(data, model))