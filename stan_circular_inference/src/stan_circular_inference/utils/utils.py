import numpy as np
import pandas as pd
import arviz as az
import stan
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def _get_percentiles(param, confidence_interval = 2.5, round_to=2):
  """
  Auxiliary function to compute percentiles.
  
  Parameters
  - `param` (array like or pd.Series): the parameter samples from a posterior distribution
  - `confidence_interval` (float): the desired confidence interval (default is 2.5 for
    a 95% interval)
  - `round_to` (int): number of decimals after the comma to round the results (default is 2)
  
  Aspects in consideration of the parameters:
  - Keep the lower confidence interval always to the left, which means that
  the confidence_interval parameter must be lower or equal to 50
  - If the confidence_interval is 50, it is only computed once

  Returns
    (pd.DataFrame): DataFrame with parameters as rows and HDI bounds as columns. Transposed (T) so parameters are rows and percentiles are columns.
  """
  return pd.Series({
      f'hdi_{confidence_interval}%' : np.percentile(param, confidence_interval).round(round_to),
      f'hdi_{(100 - confidence_interval)}%': np.percentile(param, 100 - confidence_interval).round(round_to)
  }) if confidence_interval != 50 else pd.Series({
      'hdi_50%': np.percentile(param, 50).round(round_to)
  })

"""
TODO: passar a ser uma função das classes e não do utils
"""
def build_model(model, model_data):
  """
  Build the model
  
  :param model: Stan model to build
  :param model_data: Model arguments
  """

  try:
    return stan.build(model, data=model_data)
  
  except Exception as e:
    if type(e) is TypeError:
      raise TypeError("Wrong parameter format for the model!")
    print(type(e))
    print("Error:", e)

def get_samples(posterior, sample_amount=50000, init=None):
  """
  Get the fit of the model
  
  :param posterior: Description
  :param sample_amount: Description
  :param init: Description
  """
  # Sample from the posterior model
  if init:
    return posterior.sample(num_chains=4, num_samples=sample_amount, init=init)
  else:
    return posterior.sample(num_chains=4, num_samples=sample_amount)

def get_values(fit, parameter):
  """
  Search for the intereset parameter values in the chains (TODO: more than 1 interest parameter)
  
  :param fit: Description
  :param parameter: Description
  """
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
  
  return interest_parameter_values

def get_statistics(fit, confidence_interval=11):
  """
  Get the ArviZ summary and costumize it with the desired confidence interval
  
  :param fit: Description
  :param confifence_interval: Description
  :param round_to: Description
  """
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
  
def get_pystan_statistics(model_data, model, parameter, confidence_interval=11, sample_amount=50000, init=None):
  posterior = build_model(model, model_data)
  fit = get_samples(posterior, sample_amount, init)
  get_statistics(fit, confidence_interval)
  return get_values(fit, parameter)

def get_nutpie_statistics(trace, confidence_interval=30):
  """
  Function utilized while testing nutpie tool.

  Same logic from the get_pystan_statistics.

  But it skips the step that builds the model.
  """
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

# Auxiliary function to do the circular graph expansion
def value_to_angle(value, min_val, max_val):
  """
  Auxiliary function that transforms the `value` in an angle of the correspondent interval between `min_val` and `max_val`

  Parameters
  - `value` (float): the value to transform in an angle
  - `min_val` (float): minimum value of the interval
  - `max_val` (float): maximum value of the interval
  """
  return value / (max_val - min_val) * 2 * np.pi

# Creates a circular graph from min_val to max_val
def circular_graphic(interest_parameter_values, n_intervals = 100, density = False, data = None, min_val = None, max_val = None):
  """
  Builds a circular graph based on the `interest_parameter_values` parameter

  Parameters
  - `interest_parameter_values` (array): the values from the chains
  - `n_intervals` (int): the number of intervals desired to equally split the values between the minimum value and maximum value
  - `density` (boolean): get a rose diagram if `False` and a circular density graph if `True`
  - `data` (array): the entire dataset to get the minimum and maximum value if no `min_val` and `max_val` is passed
  - `min_val` (float): the minimum value to include in the graphic
  - `max_value` (float): the maximum value to include in the graphic
  """

  if not data and (min_val == None or max_val == None):
    raise ValueError("circular_graphic function need the data parameter or min_val and max_val parameters")

  # Create 'n_intervals' intervals between min_val and max_val
  if min_val == None: # cannot be "if not min_val" due to 0 value
    min_val = min(data)
  if max_val == None:
    max_val = max(data)
  
  bins = np.linspace(min_val, max_val, n_intervals + 1)

  # Calculate frequencies for each interval
  frequencies, _ = np.histogram(interest_parameter_values, bins=bins)

  # Create even angles for the 100 intervals
  angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)
  angles_closed = np.append(angles, angles[0])

  # Normalize frequencies to density [0, 1]
  density_values = frequencies / np.max(frequencies) if np.max(frequencies) > 0 else frequencies
  density_closed = np.append(density_values, density_values[0])

  # Creates the figure with polar projection
  fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

  y_max = None

  if density:
    # Adjust KDE (Kernel Density Estimation) to the data
    kde = gaussian_kde(interest_parameter_values)
    
    # Create numerous points to get a smoother graph
    n_smooth_points = 360
    angles_smooth = np.linspace(0, 2 * np.pi, n_smooth_points, endpoint=False)
    values_smooth = np.linspace(min_val, max_val, n_smooth_points)
    
    # Calculate smooth density
    density_smooth = kde(values_smooth)
    density_smooth = density_smooth / np.max(density_smooth)  # Normalizar
    
    # Close the circule
    angles_closed = np.append(angles_smooth, angles_smooth[0])
    density_closed = np.append(density_smooth, density_smooth[0])
    
    # Plot smooth density
    ax.fill(angles_closed, density_closed, alpha=0.7, color='blue', label='KDE')
    ax.plot(angles_closed, density_closed, color='darkblue', linewidth=2)

    # For density use Y limit between 0 and 1.1
    y_max = 1.1

  else:
    # Create the circular bars
    bars = ax.bar(angles, frequencies, width=2*np.pi/n_intervals,
                  align='center', alpha=0.7, edgecolor='white', linewidth=0.5)

    # Add color
    for i, bar in enumerate(bars):
        bar.set_facecolor(plt.cm.viridis(i / n_intervals))

    # For bars, use the limit Y based on frequencies
    y_max = max(frequencies) * 1.4
    # Remove default labels from the radius
    ax.set_yticklabels([])

  # Use fixed angles for labels (every 45 degrees)
  label_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

  # Calculate what values correspond to these fixed angles
  labels = []
  for pos_angle in label_angles:
    # Converts the angle to a real value
    real_value = min_val + (pos_angle / (2 * np.pi)) * (max_val - min_val)
    real_angle_deg = np.rad2deg(real_value) % 360
    labels.append(f'{real_value:.3f}\n({real_angle_deg:.0f}°)')

  # Set graphic labels
  ax.set_xticks(label_angles)
  ax.set_xticklabels(labels, fontsize=8)

  # Graphic configurations
  ax.set_theta_offset(np.pi/2)  # Starts at the top (0° at the top)
  ax.set_theta_direction(-1)    # Clock-wise
  ax.set_ylim(0, y_max)         # Space for the labels
  ax.grid(True, alpha=0.3)

  plt.title(f'Diagrama de rosas')
  plt.show()