import numpy as np
import pandas as pd
import arviz as az
import stan
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.service.data_type import DataType

class BayesianInferenceService:

  def __init__(self, model):
    if isinstance(model, ProbabilisticModel):
      self.model = model.gen_stan_model()
    else:
      self.model = model

  def __get_percentiles(self, param, confidence_interval = 2.5, round_to=2):
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

  def build_model(self, data):
    """
    Build the model
    
    :param model: Stan model to build
    :param model_data: Model arguments
    """

    try:
      return stan.build(self.model, data=data)
    
    except Exception as e:
      if type(e) is TypeError:
        raise TypeError("Wrong parameter format for the model!")
      if type(e) is RuntimeError:
        raise RuntimeError("Try specifying initial values, reducing ranges of constrained values, reparameterizing the model.")
      if type(e) is TimeoutError:
        raise TimeoutError("The model timeout during sampling, try reducing the sampling amount!")
      print(type(e))
      print("Error:", e)

  def get_samples(self, posterior, sample_amount=50000, init=None):
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
      res = posterior.sample(num_chains=4, num_samples=sample_amount)

      return res

  def get_values(self, fit, parameters):
    """
    Search for the intereset parameter values in the chains
    
    :param fit: Description
    :param parameters: Description

    Note: To access a vector element, you must pass as parameter:
    vector_name.1 to get the values for the first element of the vector.
    """

    vals = {}
    chains = fit.stan_outputs

    for parameter in parameters:
      vals[parameter] = []

      for chain in chains:
        lines = chain.decode('utf-8').strip().split('\n')

        for line in lines:
          data = json.loads(line)
          values = data['values']
          if data['topic'] == 'sample' and isinstance(values, dict):
            interest_parameter = values[parameter]
            
            vals[parameter].append(interest_parameter)
    
    return vals

  def get_statistics(self, fit, confidence_interval=11):
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
      percentiles = pd.DataFrame({param: self.__get_percentiles(fit[param], min_val) for param in fit.keys()}).T

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
    
  def get_pystan_statistics(self, data, parameters, confidence_interval=11, sample_amount=50000, init=None):
    posterior = self.build_model(data)
    fit = self.get_samples(posterior, sample_amount, init)
    self.get_statistics(fit, confidence_interval)
    return self.get_values(fit, parameters)

  def get_nutpie_statistics(self, trace, confidence_interval=30):
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
          param: self.__get_percentiles(posterior_df[param], min_val) for param in params
      }).T

      # Add the confidence intervals manually
      #percentiles = pd.DataFrame({param: __get_percentiles(trace[param], min_val) for param in trace.keys()}).T

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
  
  def normalize_values(self, values, min_val, max_val):
    return [((value - min_val) % (max_val - min_val)) + min_val for value in values]

  # Auxiliary function to do the circular graph expansion
  def values_to_angles(self, values, min_val, max_val):
    """
    Auxiliary function that transforms the `value` in an angle of the correspondent interval between `min_val` and `max_val`

    Parameters
    - `value` (float): the value to transform in an angle
    - `min_val` (float): minimum value of the interval
    - `max_val` (float): maximum value of the interval
    """
    return [(((value - min_val) / (max_val - min_val)) % 1.0) * (max_val - min_val) for value in values]

  # Creates a circular graph from min_val to max_val
  def circular_graphic(self, interest_parameter_values, n_intervals = 100, density = False, data = None, min_val = None, max_val = None, data_type=DataType.RADS):
    """
    Builds a circular graph based on the `interest_parameter_values` parameter

    TODO fazer para vários parâmetros e juntar(?) no mesmo gráfico

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

    normalized_values = self.normalize_values(interest_parameter_values, data_type.value[0], data_type.value[1])

    # Does the expansion on the graph, for example 40 to 60 degrees instead of the whole 360 degrees
    filtered_values = [value for value in normalized_values if min_val <= value <= max_val]

    # Calculate frequencies for each interval
    frequencies, _ = np.histogram(filtered_values, bins=bins)

    # Create even angles for the 100 intervals
    angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    # Normalize frequencies to density [0, 1]
    density_values = frequencies / np.max(frequencies) if np.max(frequencies) > 0 else frequencies
    density_closed = np.append(density_values, density_values[0])

    # Creates the figure with polar projection
    _, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    y_max = None

    if density:
      # Adjust KDE (Kernel Density Estimation) to the data
      kde = gaussian_kde(filtered_values)
      
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
      y_max = max(frequencies) * 1.1
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
  
  def match_points_to_distributions(self, values, real_attribution, inferred_attribution, n_intervals=100,
                                           data=None, min_val=None, max_val=None, data_type=DataType.RADS,
                                           colors=['red', 'blue'], labels=['Distribution 1', 'Distribution 2'],
                                           alpha_points=0.8, point_size=40, alpha_bars=0.4):
    """
    Draws a rose diagram with circular bars showing the overall distribution,
    and points showing real attribution (unfilled squares) and inferred attribution (tight plus signs)
    """
    
    # Validate inputs
    if len(values) != len(real_attribution) or len(values) != len(inferred_attribution):
        raise ValueError(f"All input arrays must have the same length")
    
    if not data and (min_val is None or max_val is None):
        raise ValueError("Need data parameter or min_val and max_val parameters")
    
    # Determine min and max values
    if min_val is None:
        min_val = min(data) if data is not None else min(values)
    if max_val is None:
        max_val = max(data) if data is not None else max(values)
    
    # Normalize values
    normalized_values = self.normalize_values(values, data_type.value[0], data_type.value[1])
    
    # Filter values within range
    filtered_indices = [i for i, val in enumerate(normalized_values) if min_val <= val <= max_val]
    filtered_values = [normalized_values[i] for i in filtered_indices]
    filtered_real = [real_attribution[i] for i in filtered_indices]
    filtered_inferred = [inferred_attribution[i] for i in filtered_indices]

    size_filtered_real = len(filtered_real)
    
    # Convert to angles
    angles = np.array(filtered_values)
    angles_amount = len(angles)
    
    # Split by REAL attribution
    angles_real_dist0 = [angles[i] for i in range(angles_amount) if filtered_real[i] == 0]
    angles_real_dist1 = [angles[i] for i in range(angles_amount) if filtered_real[i] == 1]
    
    # Split by INFERRED attribution
    angles_inferred_dist0 = [angles[i] for i in range(angles_amount) if filtered_inferred[i] == 0]
    angles_inferred_dist1 = [angles[i] for i in range(angles_amount) if filtered_inferred[i] == 1]
    
    # Count points
    count_total = len(filtered_values)
    count_real_dist0 = len(angles_real_dist0)
    count_real_dist1 = len(angles_real_dist1)
    count_inferred_dist0 = len(angles_inferred_dist0)
    count_inferred_dist1 = len(angles_inferred_dist1)
    
    # Calculate accuracy
    correct_predictions = sum(1 for i in range(size_filtered_real) if filtered_real[i] == filtered_inferred[i])
    accuracy = correct_predictions / count_total * 100 if count_total > 0 else 0
    
    # Create figure with extra space at top and bottom
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    
    # ----------------------------------------------------------------------
    # PART 1: Add circular bars
    # ----------------------------------------------------------------------
    # Create bins for the bars
    bins = np.linspace(min_val, max_val, n_intervals + 1)
    
    # Calculate frequencies for all values
    frequencies, _ = np.histogram(angles, bins=bins)
    
    # Create even angles for the intervals
    bar_angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)
    
    # Create the circular bars with transparency
    bars = ax.bar(bar_angles, frequencies, width=2*np.pi/n_intervals,
                  align='center', alpha=alpha_bars, edgecolor='white', 
                  linewidth=0.5)
    
    # Add color to bars
    for bar in bars:
        bar.set_facecolor(plt.cm.viridis(0.5))  # Single color for bars
    
    # Get y_max from bars for scaling
    y_max_bars = max(frequencies) * 1.1 if len(frequencies) > 0 else 1.0
    
    # ----------------------------------------------------------------------
    # PART 2: Add points
    # ----------------------------------------------------------------------
    # Place points at a fixed radius inside the bars
    point_radius = y_max_bars * 1.05  # Place points at 95% of max bar height
    
    # Real attribution - UNFILLED SQUARES
    if count_real_dist0 > 0:
        ax.scatter(angles_real_dist0, np.ones(count_real_dist0) * point_radius, 
                  c='none', edgecolors=colors[0], s=point_size*1.5, 
                  alpha=alpha_points, marker='s', linewidths=1.5,
                  label=f'REAL {labels[0]} (n={count_real_dist0})', zorder=3)
    
    if count_real_dist1 > 0:
        ax.scatter(angles_real_dist1, np.ones(count_real_dist1) * point_radius, 
                  c='none', edgecolors=colors[1], s=point_size*1.5,
                  alpha=alpha_points, marker='s', linewidths=1.5,
                  label=f'REAL {labels[1]} (n={count_real_dist1})', zorder=3)
    
    # Inferred attribution - TIGHT PLUS SIGNS
    if count_inferred_dist0 > 0:
        ax.scatter(angles_inferred_dist0, np.ones(count_inferred_dist0) * point_radius, 
                  c=colors[0], s=point_size*0.7, alpha=alpha_points, 
                  marker='+', linewidths=2.0,
                  label=f'INFERRED {labels[0]} (n={count_inferred_dist0})', zorder=4)
    
    if count_inferred_dist1 > 0:
        ax.scatter(angles_inferred_dist1, np.ones(count_inferred_dist1) * point_radius, 
                  c=colors[1], s=point_size*0.7, alpha=alpha_points, 
                  marker='+', linewidths=2.0,
                  label=f'INFERRED {labels[1]} (n={count_inferred_dist1})', zorder=4)
    
    # ----------------------------------------------------------------------
    # PART 3: Set up the plot
    # ----------------------------------------------------------------------
    # Set up the polar plot
    ax.set_theta_offset(np.pi/2)  # Start at top
    ax.set_theta_direction(-1)    # Clockwise
    
    # Set radial limit - keep it tight to the data
    y_max = y_max_bars * 1.1  # Just a little space at the top
    ax.set_ylim(0, y_max)
    
    # Hide radial labels
    ax.set_yticklabels([])
    
    # X-tick labels (every 45 degrees)
    label_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    tick_labels = []
    
    for pos_angle in label_angles:
        real_value = min_val + (pos_angle / (2 * np.pi)) * (max_val - min_val)
        real_angle_deg = np.rad2deg(real_value) % 360
        tick_labels.append(f'{real_value:.2f}\n({real_angle_deg:.0f}°)')
    
    ax.set_xticks(label_angles)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # ----------------------------------------------------------------------
    # PART 4: Add text elements using figure coordinates (NO overlap)
    # ----------------------------------------------------------------------
    # Main title at the very top of the figure
    fig.suptitle('Rose Diagram with Attribution Points', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Subtitle with legend explanation
    fig.text(0.5, 0.92, 'Bars: Overall Distribution | □ Real | + Inferred', 
            ha='center', fontsize=12, color='gray')
    
    # Confusion matrix calculation
    confusion_matrix = {
        'true_0_pred_0': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 0 and filtered_inferred[i] == 0),
        'true_0_pred_1': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 0 and filtered_inferred[i] == 1),
        'true_1_pred_0': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 1 and filtered_inferred[i] == 0),
        'true_1_pred_1': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 1 and filtered_inferred[i] == 1)
    }
    
    # Calculate per-class accuracy
    accuracy_dist0 = confusion_matrix['true_0_pred_0'] / (confusion_matrix['true_0_pred_0'] + confusion_matrix['true_0_pred_1']) * 100 if (confusion_matrix['true_0_pred_0'] + confusion_matrix['true_0_pred_1']) > 0 else 0
    accuracy_dist1 = confusion_matrix['true_1_pred_1'] / (confusion_matrix['true_1_pred_1'] + confusion_matrix['true_1_pred_0']) * 100 if (confusion_matrix['true_1_pred_1'] + confusion_matrix['true_1_pred_0']) > 0 else 0
    
    # Main statistics at the bottom
    fig.text(0.5, 0.08,  # Position at bottom
             f'Total: {count_total} points | Accuracy: {accuracy:.1f}%',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    # Distribution counts
    fig.text(0.5, 0.04,  # Below the main stats
             f'{labels[0]}: Real={count_real_dist0}, Inferred={count_inferred_dist0} | '
             f'{labels[1]}: Real={count_real_dist1}, Inferred={count_inferred_dist1}',
             ha='center', fontsize=10)
    
    # Per-class accuracy at the very bottom
    fig.text(0.5, 0.01,  # At the very bottom
             f'Class Accuracy: {labels[0]}={accuracy_dist0:.1f}%, {labels[1]}={accuracy_dist1:.1f}%',
             ha='center', fontsize=9, style='italic')
    
    # Legend - place it to the right to avoid overlap
    if count_total > 0:
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), 
                          fontsize=9, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('lightgray')
    
    # Adjust layout to make room for all the text
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0, right=0.95)
    plt.show()
    
    return {
        'total_points': count_total,
        'accuracy': accuracy,
        'accuracy_dist0': accuracy_dist0,
        'accuracy_dist1': accuracy_dist1,
        'correct_predictions': correct_predictions,
        'confusion_matrix': confusion_matrix,
        'frequencies': frequencies,
        'bar_angles': bar_angles,
        'real': {
            'distribution0': count_real_dist0,
            'distribution1': count_real_dist1
        },
        'inferred': {
            'distribution0': count_inferred_dist0,
            'distribution1': count_inferred_dist1
        }
    }