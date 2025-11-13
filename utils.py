import numpy as np
import pandas as pd
import arviz as az
import stan
import json
import matplotlib.pyplot as plt

# Auxiliary function to compute percentiles
# parameters:
# - param: array object, the parameter samples from which to compute the percentiles
# - confidence_interval: int, the desired confidence interval (default is 2.5 for
#   a 95% interval)
# - round_to: int, number of decimals after the comma to round the results
# Aspects in consideration of the parameters:
# - Keep the lower confidence interval always to the left, which means that
# the confidence_interval parameter must be lower or equal to 50;
# - If the confidence_interval is 50, it is only computed once;
def _get_percentiles(param, confidence_interval = 2.5, round_to=2):
  return pd.Series({
    f'hdi_{confidence_interval}%' : np.percentile(param, confidence_interval).round(round_to),
    f'hdi_{(100 - confidence_interval)}%': np.percentile(param, 100 - confidence_interval).round(round_to)
  }) if confidence_interval != 50 else pd.Series({
    'hdi_50%': np.percentile(param, 50).round(round_to)
  })

# TODO
def get_pystan_statistics(model_data, model, parameter, confidence_interval=11, sample_amount=100000, init=None):
  # Build the model
  posterior = stan.build(model, data=model_data)

  # Sample from the posterior model
  fit = None
  if init:
    fit = posterior.sample(num_chains=4, num_samples=sample_amount, init=init)
  else:
    fit = posterior.sample(num_chains=4, num_samples=sample_amount)

  interest_parameter_values = []
  chains = fit.stan_outputs

  for chain in chains:
    lines = chain.decode('utf-8').strip().split('\n')

    for line in lines:
      data = json.loads(line)
      values = data['values']
      if data['topic'] == 'sample' and isinstance(values, dict):
        interest_parameter = values[parameter]
        
        interest_parameter_values.append(interest_parameter)

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

    # This ensures that the lower confidence interval is always to the left
    min_val = min(confidence_interval, 100 - confidence_interval)

    # Add the confidence intervals manually
    percentiles = pd.DataFrame({param: _get_percentiles(fit[param], min_val) for param in fit.keys()}).T

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

  return interest_parameter_values

# trace == az_data => ArviZ InferenceData object
def get_nutpie_statistics(trace, confidence_interval=30):
  # First get the default summary (includes mean, sd, ess, r_hat)
  summary_df = az.summary(trace, round_to=2)

  # Initialize the minimum value for the confidence interval
  min_val = None
  # Initialize manual percentiles variable
  percentiles = None

  # If the confidence interval is different from the default 3%
  if confidence_interval != 3:

    # This ensures that the lower confidence interval is always to the left
    min_val = min(confidence_interval, 100 - confidence_interval)

    # Convert the posterior samples to a DataFrame so we can easily access the columns
    posterior_df = trace.posterior.to_dataframe()

    # Get only the parameter columns
    params = [col for col in posterior_df.columns if col not in ["chain", "draw"]]

    # Add the confidence intervals manually for each parameter
    percentiles = pd.DataFrame({
      param: _get_percentiles(posterior_df[param], min_val) for param in params
    }).T

    # Add the confidence intervals manually
    #percentiles = pd.DataFrame({param: _get_percentiles(trace[param], min_val) for param in trace.keys()}).T

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
  
def circular_graphic(interest_parameter_values, data, n_intervals = 100, min_value = None, max_value = None):
    
  # Create 'n_intervals' intervals between min_val and max_val
  if not min_value:
    min_value = min(data)
  if not max_value:
    max_value = max(data)
      
  bins = np.linspace(min_value, max_value, n_intervals + 1)

  # Calculate frequencies for each interval
  frequencies, _ = np.histogram(interest_parameter_values, bins=bins)

  # Create even angles for the 100 intervals
  angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)

  # Creates the figure with polar projection
  fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

  # Create the circular bars
  bars = ax.bar(angles, frequencies, width=2*np.pi/n_intervals,
                align='center', alpha=0.7, edgecolor='white', linewidth=0.5)

  # Add color gradient
  for i, bar in enumerate(bars):
    bar.set_facecolor(plt.cm.viridis(i / n_intervals))

  # Add the interval values around the circunference
  for i, (angle, freq) in enumerate(zip(angles, frequencies)):
    if freq > 0:  # Add only values to intervals with frequency > 0
      label_radius = max(frequencies) * 1.3
      # Show the middle point of the interval
      bin_center = (bins[i] + bins[i+1]) / 2
      ax.text(angle, label_radius, f"{bin_center:.1f}", 
              ha='center', va='center', fontsize=6)

  # Graphic configurations
  ax.set_theta_offset(np.pi/2)  # Começar do topo (0° no topo)
  ax.set_theta_direction(-1)    # Sentido horário
  ax.set_ylim(0, max(frequencies) * 1.4)  # Espaço para os labels
  ax.grid(True, alpha=0.3)

  # Remove labels from the radius
  ax.set_yticklabels([])

  plt.title(f'Diagrama de rosas')
  plt.show()