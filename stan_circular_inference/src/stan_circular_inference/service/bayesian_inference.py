import numpy as np
import arviz as az
import stan
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.service.data_type import DataType
import math
import pandas as pd
import re

class BayesianInferenceService:
  """
  Service class for Bayesian inference using Stan, with special support for circular data.

  This class provides a complete pipeline for building Stan models, sampling from
  posterior distributions, extracting and summarising results, and visualising
  both linear and circular parameters. It also includes specialised methods for
  evaluating mixture models (e.g., component attribution accuracy and visualisation).

  The class is designed to work with PyStan and integrates with ArviZ for summary 
  statistics and diagnostics. Circular parameters (identified by names starting with 'mu')
  are automatically handled: means and HDI bounds are wrapped to the [0, 2π) interval.
  
  Main workflow:
    1. Build a Stan model using `build_model()`.
    2. Draw posterior samples with `get_samples()`.
    3. Obtain a summary DataFrame via `get_statistics()`.
    4. Extract raw posterior values for specific parameters using `get_values()`.
    5. Visualise distributions with `circular_graphic()` or `multiple_graphics()`.

    For mixture models, additional methods help assess classification accuracy:
      - `get_mixture_statistics()` computes confusion matrix and accuracies.
      - `show_mixture_statistics()` prints the results.
      - `match_points_to_distributions()` creates a polar plot comparing true vs inferred labels.

    Parameters
    ----------
    model : ProbabilisticModel
        An instance of a probabilistic model (e.g., VonMisesMK, Mixture) that provides
        the Stan code and prior specifications. Or a Stan model provided

    Attributes
    ----------
    model : ProbabilisticModel
        The model object used for generating Stan code.

    Methods
    -------
    build_model(data)
        Compile and build the Stan model with the provided data.
    get_samples(posterior, sample_amount=50000, init=None)
        Draw MCMC samples from the posterior distribution.
    get_values(fit, parameters)
        Extract raw posterior values for specified parameters.
    get_statistics(fit, confidence_interval=89)
        Generate a summary DataFrame with posterior statistics and HDI.
    get_pystan_statistics(data, parameters, confidence_interval=11,
                          sample_amount=50000, init=None)
        Convenience method that runs the full pipeline (build, sample, summarise, extract).
    normalize_values(values, min_val, max_val)
        Normalise a list of values into a target interval using modulo arithmetic.
    circular_graphic(interest_parameter_values, n_intervals=100, density=False,
                     data=None, min_val=None, max_val=None, data_type=DataType.RADS)
        Create a rose diagram or circular density plot for a single parameter.
    multiple_graphics(interest_parameter_values, ...)
        Create a grid of subplots (circular or linear) for multiple parameters.
    get_mixture_statistics(values, real_attribution, inferred_attribution,
                           min_val=None, max_val=None, data_type=DataType.PERCENT)
        Compute classification accuracy and confusion matrix for mixture assignments.
    show_mixture_statistics(mixture_statistics)
        Print mixture evaluation statistics in a formatted table.
    match_points_to_distributions(values, real_attribution, inferred_attribution, ...)
        Create a polar plot comparing true and inferred component labels for mixture models.
    _draw_circular_subplot(ax, filtered_values, param_min_val, param_max_val,
                           n_intervals, density, show_values, value_format, param_name)
        Helper to draw a circular subplot (used by `multiple_graphics`).
    _draw_linear_subplot(ax, filtered_values, param_min_val, param_max_val,
                         param_name, n_intervals=20)
        Helper to draw a linear subplot (used by `multiple_graphics`).
  
  Examples
  --------
  >>> from stan_circular_inference.models import VonMisesMK
  >>> from stan_circular_inference.priors import Uniform, Exponential
  >>> model = VonMisesMK(Uniform(0, 2*np.pi), Exponential(0.1))
  >>> service = BayesianInferenceService(model)
  """

  def __init__(self, model):
    if isinstance(model, ProbabilisticModel):
      self.model = model.gen_stan_model()
    # In case the user provide it's own Stan model via `str`
    else:
      self.model = model

    if 'functions' not in self.model and self.__dist_in_model():
        self.model = """
        functions{
        
        }
        """ + self.model[:]
    
    str_idx = self.model.find('functions')
    par_idx = self.model.find('{', str_idx) + 1
    
    if self.model.find('~ cardioid_lpdf') != -1:
        self.model = self.model[:par_idx] + """
        real cardioid_lpdf(real value, real mu, real rho) {
            return -log(2*pi()) + log1p(2*rho*cos(value - mu));
        }
        """ + self.model[par_idx + 1:]
    if self.model.find('~ cardioid') != -1:
        self.model = self.model[:par_idx] + """
        real cardioid_lpdf(real value, real mu, real rho) {
            return -log(2*pi()) + log1p(2*rho*cos(value - mu));
        }
        """ + self.model[par_idx + 1:]
        self.model = self.__replace_sampling_statement(self.model, 'cardioid')
    if self.model.find('~ wrapped_cauchy_lpdf') != -1:
        self.model = self.model[:par_idx] + """
        real wrapped_cauchy_lpdf(real value, real mu, real rho) {
            real rho_sqr = square(rho);
            return -log(2*pi()) + log1m(rho_sqr) - log1p(rho_sqr - 2*rho*cos(value - mu));
        }
        """ + self.model[par_idx + 1:]
    if self.model.find('~ wrapped_cauchy') != -1:
        self.model = self.model[:par_idx] + """
        real wrapped_cauchy_lpdf(real value, real mu, real rho) {
            real rho_sqr = square(rho);
            return -log(2*pi()) + log1m(rho_sqr) - log1p(rho_sqr - 2*rho*cos(value - mu));
        }
        """ + self.model[par_idx + 1:]
        self.model = self.__replace_sampling_statement(self.model, 'wrapped_cauchy')

  def __replace_sampling_statement(self, model_code: str, dist_name: str) -> str:
    """
    Replace sampling statements of the form:
        lhs ~ dist_name(param1, param2);
    with:
        target += dist_name_lpdf(lhs | param1, param2);

    Assumes exactly two parameters inside parentheses.
    """
    # Pattern matches any whitespace between tokens
    pattern = rf'(\S+)\s*~\s*{re.escape(dist_name)}\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*;'
    
    def repl(match):
        lhs = match.group(1)
        param1 = match.group(2).strip()
        param2 = match.group(3).strip()
        return f'target += {dist_name}_lpdf({lhs} | {param1}, {param2});'
    
    return re.sub(pattern, repl, model_code)
  
  def __dist_in_model(self):
      return 'cardioid' in self.model or 'cardioid_lpdf' in self.model or 'wrapped_cauchy' in self.model or 'wrapped_cauchy_lpdf' in self.model

  def build_model(self, data:dict) -> stan.model:
    """
    Compile and build the Stan model with the provided data.
    
    Parameters
    ----------
    data : dict
        A dictionary where keys correspond to data variables declared in the
        Stan model (e.g., `{'N': 20, 'values': np.array([...])}`). Values must
        be of the appropriate types (integers, floats, arrays, etc.) as expected
        by the Stan model.

    Returns
    -------
    stan.model
        A compiled Stan model object that can be used for sampling (e.g., with
        `sample()`).

    Raises
    ------
    TypeError
        If the provided `data` dictionary does not match the expected format
        (e.g., missing keys, wrong data types). Wraps the original `TypeError`
        from `stan.build`.

    RuntimeError
        If the model fails to build due to numerical or structural issues.
        Suggested remedies include: providing initial values, reducing the
        ranges of constrained parameters, or reparameterizing the model.

    Example
    --------
    >>> model = '''
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
        } '''
    >>> data = {'N': 100, 'values': np.random.randn(100)}
    >>> service = BayesianInferenceService(model)
    >>> posterior = service.build_model(data)
    """

    try:
        return stan.build(self.model, data=data)
    
    except Exception as e:
        if type(e) is TypeError:
            raise TypeError("Wrong parameter format for the model!")
        if type(e) is RuntimeError:
            raise RuntimeError("Try specifying initial values, reducing ranges of constrained values, reparameterizing the model.")

  # adicionar testes para esta fun sobre o timeouterror
  def get_samples(self, posterior:stan.model, sample_amount:int=50000, n_chains:int=4, init=None):
    """
    Draw samples from the posterior distribution using MCMC.
    
    Parameters
    ----------
    posterior : stan.model
        A compiled Stan model object returned by `build_model()`.
    sample_amount : int, default 50000
        Number of posterior samples to draw **per chain** after warmup.
        The total number of draws will be `n_chains * sample_amount`.
    n_chains : int, default 4
        Number of chains used to draw samples.
    init : dict or list of dicts, optional
        Initial values for the parameters. Can be:
        - A single dictionary (same initial values for all chains).
        - A list of dictionaries (one per chain).
        - `None` (default): Stan randomly generates initial values.

    Returns
    -------
    stan.StanFit
        A Stan fit object containing the posterior samples, diagnostics,
        and metadata. This object can be passed to `get_statistics()`
        or used directly for inspection.

    Raises
    ------
    TimeoutError
        If the model sampling process times out. This may indicate that the
        model is too complex or it is doing a high amount of samlples.

    Notes
    -----
    - For large `sample_amount`, sampling may take a long time. Consider
      reducing it during exploratory analysis.
    - If the model has convergence issues, try providing custom `init`
      values or adjusting the model's parameterization.

    Example
    --------
    >>> posterior = service.build_model(data)
    >>> fit = service.get_samples(posterior, sample_amount=10000)
    >>> summary = service.get_statistics(fit)
    """

    try:
        # Sample from the posterior model
        if init:
            return posterior.sample(num_chains=n_chains, num_samples=sample_amount, init=init)
        else:
            return posterior.sample(num_chains=n_chains, num_samples=sample_amount)
    except Exception:
        raise TimeoutError("The model timeout during sampling, try reducing the sampling amount or passing inital values!")

  def get_values(self, fit, parameters:list[str]) -> dict:
    """
    Extract posterior samples for specific parameters from a Stan fit object.
    This method parses the raw Stan output (text lines) to retrieve the values
    of the requested parameters from each sample of all chains.
    
    Parameters
    ----------
    fit : StanFit
        A Stan fit object (e.g., returned by `get_samples()`) that contains the
        raw output in `fit.stan_outputs` as a list of byte strings per chain.
    parameters : list of str
        Names of the parameters to extract. Parameter names must match exactly
        those used in the Stan model. For vector parameters, use the dot notation
        (see Note below).

    Returns
    -------
    dict
        A dictionary where each key is a parameter name (as given in `parameters`)
        and each value is a list of posterior samples (one entry per draw across
        all chains, in the order they appear in the raw output).

    Note
    ----
    To access individual elements of a vector parameter, you must pass the name
    with a dot followed by the element index, starting at 1. For example:
    - `"theta.1"` for the first element of vector `theta`.
    - `"beta.3"` for the third element of vector `beta`.

    Examples
    --------
    >>> fit = service.get_samples(posterior)
    >>> params = ["mu", "kappa", "mixing_prop.2"]
    >>> samples = service.get_values(fit, params)
    >>> mu_samples = samples["mu"]
    >>> kappa_samples = samples["kappa"]
    >>> mixing_second = samples["mixing_prop.2"]
    """

    vals = {p: [] for p in parameters}

    for chain in fit.stan_outputs:
        lines = chain.decode('utf-8').strip().split('\n')
        for line in lines:
            data = json.loads(line)
            if data.get('topic') != 'sample':
                continue
            values = data.get('values')
            if not isinstance(values, dict):
                continue
            for parameter in parameters:
                vals[parameter].append(values[parameter])
    
    return vals

  def get_statistics(self, fit, confidence_interval:int=89) -> pd.DataFrame:
    """
    Generate a summary DataFrame with posterior statistics for all parameters.
    
    This method converts a Stan fit object to an ArviZ InferenceData, computes
    a summary (mean, standard deviation, effective sample size, r‑hat, and HDI),
    and then adjusts circular parameters (those whose names start with `'mu'`)
    by wrapping their mean and HDI bounds into the interval [0, 2π).

    Parameters
    ----------
    fit : StanFit
        A Stan fit object returned by `get_samples()`. Must be compatible with
        `az.from_pystan()`.
    confidence_interval : int, default 89
        The highest density interval (HDI) probability as a percentage.
        For example, `95` computes a 95% HDI. The method automatically
        calculates the lower and upper HDI columns as `hdi_{x}%` and
        `hdi_{y}%` where `x = (100 - ci)/2` and `y = ci + x`.

    Returns
    -------
    pandas.DataFrame
        Summary table with rows for each model parameter and columns including:
        - `mean` (circular mean for `mu*` parameters, linear mean otherwise)
        - `sd`
        - `ess_bulk`, `ess_tail`
        - `r_hat`
        - `hdi_{x}%`, `hdi_{y}%` (wrapped to [0, 2π) for `mu*` parameters)

    Notes
    -----
    - Circular variables are identified by name starting with `'mu'` (e.g.,
      `mu`, `mu1`, `mu2`). For these, the mean and HDI bounds are taken modulo
      2π to respect circular topology.
    - The method prints the summary DataFrame to the console before returning it.
    - Rounding is fixed to 5 decimal places.

    Examples
    --------
    >>> fit = service.get_samples(posterior)
    >>> summary = service.get_statistics(fit, confidence_interval=95)
    >>> print(summary.loc['mu1', 'mean'])
    >>> print(summary.loc['mu1', 'hdi_2.5%'], summary.loc['mu1', 'hdi_97.5%'])
    """

    # Convert to ArviZ InferenceData object
    az_data = az.from_pystan(fit)

    # Identify all keys that are circular variables
    all_vars = az_data.posterior.data_vars
    circ_vars = [var for var in all_vars if var.startswith('mu')]

    # First get the default summary (includes mean, sd, ess, r_hat)
    summary_df = az.summary(
        az_data,
        round_to = 5,
        circ_var_names = circ_vars,
        hdi_prob=confidence_interval / 100
    )

    min_hdi = (100 - confidence_interval) / 2
    max_hdi = confidence_interval + min_hdi

    mean_col = 'mean'
    hdi_low_col = f'hdi_{min_hdi}%'
    hdi_high_col = f'hdi_{max_hdi}%'

    for param in circ_vars:

        summary_df.loc[param, mean_col] = summary_df.loc[param, mean_col] % (2 * np.pi)

        summary_df.loc[param, hdi_low_col] = summary_df.loc[param, hdi_low_col] % (2 * np.pi)
        summary_df.loc[param, hdi_high_col] = summary_df.loc[param, hdi_high_col] % (2 * np.pi)

    print(summary_df)

    return summary_df
  
  def get_pystan_statistics(self, data:dict, parameters:list[str], confidence_interval:int=11, sample_amount:int=50000, n_chains:int=4, init=None) -> dict:
    """
    Run a full PyStan inference pipeline and return posterior samples for specified parameters.

    This method sequentially:
    1. Builds the Stan model with the provided data.
    2. Draws MCMC samples from the posterior.
    3. Generates a summary DataFrame (including means, SDs, HDIs, and diagnostics).
    4. Extracts and returns the raw posterior values for the requested parameters.
 
    Parameters
    ----------
    data : dict
        A dictionary of data variables required by the Stan model (e.g.,
        `{'N': 20, 'values': np.array([...])}`).
    parameters : list of str
        Names of the parameters (or transformed parameters) to extract from the
        posterior. For vector elements, use dot notation (e.g., `'mu.1'`).
    confidence_interval : int, default 11
        The highest density interval (HDI) probability expressed as a percentage.
        This is passed to `get_statistics()`. The default 11% is unusually low;
        you may want to increase it (e.g., 89 or 95) for most use cases.
    sample_amount : int, default 50000
        Number of posterior samples to draw **per chain** (after warmup). The total
        number of draws will be `num_chains * sample_amount`. The number of chains
        is fixed inside `get_samples()` (typically 4).
    init : dict or list of dicts, optional
        Initial values for the parameters. Passed directly to `get_samples()`.
        If `None`, Stan generates random initial values.

    Returns
    -------
    dict
        A dictionary where each key is a parameter name (from `parameters`) and each
        value is a list of posterior samples (one entry per draw across all chains,
        in the order they appear in the raw output). The structure is identical to
        that returned by `get_values()`.

    Examples
    --------
    >>> data = {'N': 100, 'values': np.random.vonmises(mu=0, kappa=2, size=100)}
    >>> params = ['mu', 'kappa', 'mixing_weight.1']
    >>> raw_samples = service.get_pystan_statistics(data, params, confidence_interval=95)
    >>> mu_samples = raw_samples['mu']
    >>> first_mixing_weight = raw_samples['mixing_weight.1']
    """
    posterior = self.build_model(data)
    fit = self.get_samples(posterior, sample_amount, n_chains, init)
    self.get_statistics(fit, confidence_interval)
    return self.get_values(fit, parameters)
  
  def normalize_values(self, values, min_val, max_val):
    """
    Normalize a list of values into a specified interval using modulo arithmetic.
    
    Parameters
    ----------
    values : list or numpy.ndarray
        A sequence of numeric values to be normalized.
    min_val : float
        The lower bound of the target interval (inclusive).
    max_val : float
        The upper bound of the target interval (exclusive for modulo operations,
        but the result will lie in `[min_val, max_val)`; if the result equals
        `max_val`, it wraps to `min_val`).

    Returns
    -------
    list
        A new list where each element is normalized into the range
        `[min_val, max_val)`.

    Notes
    -----
    - The function uses the modulo operator `%` which in Python always returns
      a non‑negative result, ensuring the output stays within the desired interval.
    - For circular data, typical usage is `normalize_values(angles, 0, 2*np.pi)`.
    - If `max_val - min_val` is the range length, the operation effectively
      computes: `((value - min_val) % length) + min_val`.
    - The function does **not** modify the original input; it returns a new list.

    Examples
    --------
    >>> angles = [4.5, 6.2, -0.3, 2*np.pi + 0.1]
    >>> normalized = normalize_values(angles, 0, 2*np.pi)
    >>> print([round(a, 2) for a in normalized])
    [4.5, 6.2, 6.0, 0.1]   # (2π ≈ 6.283, so 6.2 stays, -0.3 wraps to 6.0)

    >>> values = [10, 15, 20]
    >>> normalize_values(values, 0, 10)
    [0, 5, 0]
    """
    return [((value - min_val) % (max_val - min_val)) + min_val for value in values]

  def circular_graphic(self, interest_parameter_values, n_intervals:int=100, density:bool=False, min_val=None, max_val=None, data_type=DataType.RADS):
    """
    Create a circular (polar) histogram or density plot for a set of angular values.

    Parameters
    ----------
    interest_parameter_values : array-like
        Raw parameter values (e.g., MCMC posterior samples) to be plotted.
    n_intervals : int, default 100
        Number of equally spaced bins/angles for the histogram or the number of
        points used for density estimation.
    density : bool, default False
        If `False`, produces a rose diagram (circular bar chart). If `True`,
        produces a smoothed circular density plot using Gaussian KDE.
    min_val : float, optional
        Minimum value (in radians) to include in the plot. If `None`, taken from
        `data` (minimum of the provided dataset). Used to filter the values.
    max_val : float, optional
        Maximum value (in radians) to include in the plot. If `None`, taken from
        `data` (maximum of the provided dataset).
    data_type : DataType, default DataType.RADS
        An enumeration defining the expected range of the input values.
        Typically `DataType.RADS` corresponds to `[0, 2π)`.

    Returns
    -------
    None
        The function displays a matplotlib polar plot and does not return a value.

    Examples
    --------
    >>> # Rose diagram for posterior samples of a mean angle
    >>> samples = np.random.vonmises(mu=np.pi, kappa=2, size=1000)
    >>> service.circular_graphic(samples, n_intervals=36, density=False)
    """

    # Create 'n_intervals' intervals between min_val and max_val
    if min_val == None: # cannot be "if not min_val" due to 0 value
        min_val = min(interest_parameter_values)
    if max_val == None:
        max_val = max(interest_parameter_values)
    
    bins = np.linspace(min_val, max_val, n_intervals + 1)

    normalized_values = self.normalize_values(interest_parameter_values, data_type.value[0], data_type.value[1])

    # Does the expansion on the graph, for example 40 to 60 degrees instead of the whole 360 degrees
    filtered_values = [value for value in normalized_values if min_val <= value <= max_val]

    # Calculate frequencies for each interval
    frequencies, _ = np.histogram(filtered_values, bins=bins)

    # Create even angles for the 100 intervals
    angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)

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

    plt.title('Diagrama de rosas')
    plt.show()

  def get_mixture_statistics(self, values, real_attribution:list[int], inferred_attribution:list[int], min_val=None, max_val=None, data_type=DataType.RADS):
    """
    Evaluate the accuracy of component assignments in a mixture model.

    Parameters
    ----------
    values : array-like
        The numerical values associated with each observation (e.g., the data points
        themselves or posterior means). Used for filtering observations that fall
        inside a specified interval.
    real_attribution : array-like of int
        True component labels (0 - 1st distribution or 1 - 2nd distribution) for each observation.
    inferred_attribution : array-like of int
        Labels inferred by the mixture model (0 - 1st distribution or 1 - 2nd distribution) for each observation.
    min_val : float, optional
        Lower bound of the interval used to filter observations (after normalization).
        If `None`, the minimum of `values` is used.
    max_val : float, optional
        Upper bound of the interval used to filter observations (after normalization).
        If `None`, the maximum of `values` is used.
    data_type : DataType, default DataType.PERCENT
        An enumeration defining the expected range of the `values`. Typically
        `DataType.PERCENT` corresponds to `[0, 100]`, but other ranges (e.g.,
        `DataType.RADS` for `[0, 2π)`) can be used. The values are normalised
        to the interval `[data_type.value[0], data_type.value[1]]` before filtering.

    Returns
    -------
    dict
        A dictionary containing:
        - `'accuracy'` : float – Overall classification accuracy (percentage of
          correctly labelled observations among all observations, not just those
          filtered).
        - `'accuracy_dist_1'` : float – Accuracy for class 0 (true label 0) defined
          as `TP / (TP + FP) * 100`, where TP = correctly labelled 0, FP = mislabelled 0.
        - `'accuracy_dist_2'` : float – Accuracy for class 1 (true label 1) defined
          as `TN / (TN + FN) * 100`.
        - `'confusion_matrix'` : dict – Contains `'TP'`, `'FN'`, `'FP'`, `'TN'`
          counts (only for the filtered observations).
    
    Raises
    ------
    ValueError
        If the lengths of `values`, `real_attribution`, and `inferred_attribution`
        are not equal.
    
    Examples
    --------
    >>> values = np.random.randn(100)
    >>> real = np.random.choice([0, 1], size=100)
    >>> inferred = np.random.choice([0, 1], size=100)
    >>> stats = service.get_mixture_statistics(values, real, inferred, min_val=0, max_val=1)
    """
    
    # Validate inputs
    if len(values) != len(real_attribution) or len(values) != len(inferred_attribution):
        raise ValueError("All input arrays must have the same length")

    # Determine min and max values
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    
    # Normalize values
    normalized_values = self.normalize_values(values, data_type.value[0], data_type.value[1])
    
    # Filter values within range
    filtered_indices = [i for i, val in enumerate(normalized_values) if min_val <= val <= max_val]
    filtered_real = [real_attribution[i] for i in filtered_indices]
    filtered_inferred = [inferred_attribution[i] for i in filtered_indices]

    size_filtered_real = len(filtered_real)
    
    # Calculate accuracy
    correct_predictions = sum(1 for i in range(size_filtered_real) if filtered_real[i] == filtered_inferred[i])
    accuracy = correct_predictions / size_filtered_real * 100 if size_filtered_real > 0 else 0

    # Confusion matrix calculation
    confusion_matrix = {
        'TP': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 0 and filtered_inferred[i] == 0),
        'FN': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 1 and filtered_inferred[i] == 0),
        'FP': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 0 and filtered_inferred[i] == 1),
        'TN': sum(1 for i in range(size_filtered_real) if filtered_real[i] == 1 and filtered_inferred[i] == 1)
    }
    
    # Calculate per-class accuracy
    accuracy_dist0 = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP']) * 100 if (confusion_matrix['TP'] + confusion_matrix['FP']) > 0 else 0
    accuracy_dist1 = confusion_matrix['TN'] / (confusion_matrix['TN'] + confusion_matrix['FN']) * 100 if (confusion_matrix['TN'] + confusion_matrix['FN']) > 0 else 0

    return {
        'accuracy': accuracy, 
        'accuracy_dist_1': accuracy_dist0, 
        'accuracy_dist_2': accuracy_dist1,
        'confusion_matrix': confusion_matrix
    }
  
  def show_mixture_statistics(self, mixture_statistics):
    """
    Display mixture model evaluation statistics in a human-readable format.

    Parameters
    ----------
    mixture_statistics : dict
        A dictionary containing the following keys (as returned by
        `get_mixture_statistics()`):
        - `'accuracy'` : float – Overall classification accuracy (percentage).
        - `'accuracy_dist_1'` : float – Accuracy for distribution/class 1.
        - `'accuracy_dist_2'` : float – Accuracy for distribution/class 2.
        - `'confusion_matrix'` : dict – With keys `'TP'`, `'FN'`, `'FP'`, `'TN'`
          (true/false positives/negatives).

    Returns
    -------
    None
        The method prints the statistics to the console and does not return a value.
    """

    offset = max(len(str(mixture_statistics['confusion_matrix']['TP'])), len(str(mixture_statistics['confusion_matrix']['FP'])))

    print(f"""
      Accuracy: {mixture_statistics['accuracy']}
      Distribution_1_Accuracy: {mixture_statistics['accuracy_dist_1']}
      Distribution_2_Accuracy: {mixture_statistics['accuracy_dist_2']}
      Confusion Matrix:
      ______________| Predicted Values
      Actual Values | {mixture_statistics['confusion_matrix']['TP']:>{offset}} | {mixture_statistics['confusion_matrix']['FN']}
                    | {mixture_statistics['confusion_matrix']['FP']:>{offset}} | {mixture_statistics['confusion_matrix']['TN']}
      
    """)
  
  def match_points_to_distributions(self, values, real_attribution:list[int], inferred_attribution:list[int], n_intervals:int=100,
                                           min_val=None, max_val=None, data_type=DataType.RADS,
                                           colors:list[str]=['red', 'blue'], labels:list[str]=['Distribution 1', 'Distribution 2'],
                                           alpha_points:float=0.8, point_size:int=40, alpha_bars:float=0.4):
    """
    Create a rose diagram comparing true and inferred component assignments.

    Draws a rose diagram with circular bars showing the overall distribution,
    and points showing real attribution (unfilled squares) and inferred attribution (tight plus signs)

    Parameters
    ----------
    values : array-like
        Numerical values (e.g., angles) associated with each observation.
    real_attribution : array-like of int
        True component labels (0 - 1st distribution or 1 - 2nd distribution) for each observation.
    inferred_attribution : array-like of int
        Labels inferred by the mixture model (0 - 1st distribution or 1 - 2nd distribution) for each observation.
    n_intervals : int, default 100
        Number of bins (bars) in the circular histogram.
    data : array-like, optional
        The original dataset used to determine `min_val` and `max_val` if those
        are not provided. Required when both `min_val` and `max_val` are `None`.
    min_val : float, optional
        Lower bound (in radians) of the angular range to display. If `None`,
        taken from `data` (or from `values` if `data` is also `None`).
    max_val : float, optional
        Upper bound (in radians) of the angular range to display. If `None`,
        taken from `data` (or from `values` if `data` is also `None`).
    data_type : DataType, default DataType.RADS
        Enum defining the expected range of the input `values`. Typically
        `DataType.RADS` corresponds to `[0, 2π)`.
    colors : list of str, default ['red', 'blue']
        Colors used for the two classes (class 0 and class 1) for the points.
        The first color is used for class 0, the second for class 1.
    labels : list of str, default ['Distribution 1', 'Distribution 2']
        Display names for the two components (used in legend and annotations).
    alpha_points : float, default 0.8
        Transparency (alpha) of the scatter points (both squares and plus signs).
    point_size : float, default 40
        Base size for the point markers. Squares are scaled by 1.5×, plus signs
        by 0.7× relative to this value.
    alpha_bars : float, default 0.4
        Transparency of the circular bars.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - `'total_points'` : int – Number of observations after filtering.
        - `'accuracy'` : float – Overall classification accuracy (percentage)
          computed by `get_mixture_statistics()`.
        - `'accuracy_dist0'` : float – Accuracy for class 0.
        - `'accuracy_dist1'` : float – Accuracy for class 1.
        - `'confusion_matrix'` : dict – As returned by `get_mixture_statistics()`.
        - `'real'` : dict – Counts of true labels: `{'distribution0': int, 'distribution1': int}`.
        - `'inferred'` : dict – Counts of inferred labels: `{'distribution0': int, 'distribution1': int}`.

    Raises
    ------
    ValueError
        If the lengths of `values`, `real_attribution`, and `inferred_attribution`
    
    Examples
    --------
    >>> angles = np.random.vonmises(mu=0, kappa=2, size=200)
    >>> true_labels = np.random.choice([0, 1], size=200)
    >>> inferred_labels = np.random.choice([0, 1], size=200)
    >>> result = service.match_points_to_distributions(
    ...     angles, true_labels, inferred_labels,
    ...     data=angles, n_intervals=36,
    ...     labels=['Component A', 'Component B']
    ... )
    """
    
    # Validate inputs
    if len(values) != len(real_attribution) or len(values) != len(inferred_attribution):
        raise ValueError("All input arrays must have the same length")
    
    # Determine min and max values
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    
    # Normalize values
    normalized_values = self.normalize_values(values, data_type.value[0], data_type.value[1])
    
    # Filter values within range
    filtered_indices = [i for i, val in enumerate(normalized_values) if min_val <= val <= max_val]
    filtered_values = [normalized_values[i] for i in filtered_indices]
    filtered_real = [real_attribution[i] for i in filtered_indices]
    filtered_inferred = [inferred_attribution[i] for i in filtered_indices]
    
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
    
    mixture_statistics = self.get_mixture_statistics(values, real_attribution, inferred_attribution, min_val, max_val)
    self.show_mixture_statistics(mixture_statistics)
    accuracy = mixture_statistics['accuracy']
    
    # Create figure with extra space at top and bottom
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    
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

    # Main title at the very top of the figure
    fig.suptitle('Rose Diagram with Attribution Points', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Subtitle with legend explanation
    fig.text(0.5, 0.92, 'Bars: Overall Distribution | □ Real | + Inferred', 
            ha='center', fontsize=12, color='gray')
    
    # Confusion matrix calculation
    confusion_matrix = mixture_statistics['confusion_matrix']
    
    # Calculate per-class accuracy
    accuracy_dist0 = mixture_statistics['accuracy_dist_1']
    accuracy_dist1 = mixture_statistics['accuracy_dist_2']
    
    # Main statistics at the bottom
    fig.text(0.5, 0.08,  # Position at bottom
             f'Total: {count_total} points | Accuracy: {accuracy:.1f}%',
             ha='center', fontsize=12, fontweight='bold',
             bbox={'boxstyle': 'round,pad=0.5', 'facecolor': 'lightyellow', 'alpha': 0.8})
    
    # Distribution counts
    fig.text(0.5, 0.04,  # Below the main stats
             f'{labels[0]}: Real={count_real_dist0}, Inferred={count_inferred_dist0} | '
             f'{labels[1]}: Real={count_real_dist1}, Inferred={count_inferred_dist1}',
             ha='center', fontsize=10)
    
    # Per-class accuracy at the very bottom
    fig.text(0.5, 0.01,  # At the very bottom
             f'Class Accuracy: {labels[0]}={accuracy_dist0:.1f}%, {labels[1]}={accuracy_dist1:.1f}%',
             ha='center', fontsize=9, style='italic')
    
    # Image label
    legend = ax.legend(loc='lower right', bbox_to_anchor=(1.4, 0.8))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('lightgray')
    
    # Adjust layout to make room for all the text
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0, right=0.95)
    plt.show()
    
    return {
        'total_points': count_total,
        'accuracy': accuracy,
        'accuracy_dist_1': accuracy_dist0,
        'accuracy_dist_2': accuracy_dist1,
        'confusion_matrix': confusion_matrix,
        'real': {
            'distribution0': count_real_dist0,
            'distribution1': count_real_dist1
        },
        'inferred': {
            'distribution0': count_inferred_dist0,
            'distribution1': count_inferred_dist1
        }
    }

  def multiple_graphics(self, interest_parameter_values, n_intervals:int=100, density:bool=False, 
                                min_val=None, max_val=None, param_names:list[str]=None, 
                                figsize:tuple=(14, 10), share_scale:bool=True, parameters_type:list[bool]=[]):
    """
    Create multiple subplots (circular or linear) for several parameters.

    If more than 9 parameters are provided, multiple figures are created 
    (each figure contains at most 3×3 subplots). Figures are automatically saved as PNG files.
    
    Parameters
    ----------
    interest_parameter_values : dict
        A dictionary where keys are parameter names and values are arrays of
        posterior samples (or any numeric data) to be plotted.
    n_intervals : int, default 100
        Number of bins/angles for circular plots; for linear plots this is
        currently unused.
    density : bool, default False
        If True, circular plots show a kernel density estimate; if False,
        they show a rose diagram (circular histogram). Ignored for linear plots.
    min_val : scalar, dict, or list, optional
        Lower bound(s) for the plot range. If a scalar, the same value is used
        for all parameters. If a dict, keys are parameter names. If a list,
        must match the number of parameters in order.
    max_val : scalar, dict, or list, optional
        Upper bound(s) for the plot range. Format same as `min_val`.
    param_names : list of str, optional
        Names of the parameters (used for titles). If `None`, keys of
        `interest_parameter_values` are used.
    figsize : tuple, default (14, 10)
        Size of each figure (width, height) in inches.
    share_scale : bool, default True
        If True, all **circular** subplots share the same radial axis limit
        (the maximum y‑limit among them). Linear plots are not affected.
    parameters_type : list of bool, default []
        A list of the same length as the number of parameters, where `True`
        indicates a circular plot and `False` a linear bar plot. If the list
        is empty or shorter than the number of parameters, all are assumed
        circular (`True`).

    Returns
    -------
    None
        The method displays the plots and saves them as `image1.png`,
        `image2.png`, etc. It does not return any value.

    Raises
    ------
    ValueError
        If the lengths of `data`, `min_val`, `max_val` (when provided as lists)
        do not match the number of parameters, or if `parameters_type` is longer
        than the number of parameters.

    Examples
    --------
    service.multiple_graphics(
    ...     interest_parameter_values=posterior,
    ...     param_names=['mu', 'kappa'],
    ...     parameters_type=[True, False],
    ...     min_val={'mu': 0, 'kappa': 0},
    ...     max_val={'mu': 2*np.pi, 'kappa': 5},
    ...     data=posterior
    ... )
    """
    
    # If no param_names parameter is passed, the dict keys are used
    if param_names is None:
        param_names = list(interest_parameter_values.keys())
    
    interest_parameter_values_list = list(interest_parameter_values.values())
    
    n_params = len(interest_parameter_values_list)
    
    # Check if the parameters_type size is correct
    if len(parameters_type) != n_params:
        # não colocar tudo a True mas sim encher com True
        parameters_type = [True] * n_params  # Circular graph by default

    # Handle min_val (can be dict, list, or single value)
    if min_val is not None:
        if isinstance(min_val, dict):
            min_val_list = [min_val.get(name, None) for name in param_names]
        elif isinstance(min_val, (list, tuple)):
            min_val_list = min_val
            if len(min_val_list) != n_params:
                raise ValueError(f"min_val length ({len(min_val_list)}) must match number of parameters ({n_params})")
        else:
            min_val_list = [min_val] * n_params
    else:
        min_val_list = [None] * n_params
    
    # Handle max_val (can be dict, list, or single value)
    if max_val is not None:
        if isinstance(max_val, dict):
            max_val_list = [max_val.get(name, None) for name in param_names]
        elif isinstance(max_val, (list, tuple)):
            max_val_list = max_val
            if len(max_val_list) != n_params:
                raise ValueError(f"max_val length ({len(max_val_list)}) must match number of parameters ({n_params})")
        else:
            max_val_list = [max_val] * n_params
    else:
        max_val_list = [None] * n_params
    
    graph_size = min(3, math.ceil(math.sqrt(n_params)))
    
    # Create the main figure to add sub-figures
    fig, axes = plt.subplots(graph_size, graph_size, figsize=figsize, squeeze=False)
    
    axes_flat = axes.flatten()
    
    # Store max values for shared scale (apenas para circulares)
    all_y_max = []

    counter = 0
    image_counter = 1
    
    # Process each parameter
    for _, (param_values, param_min_val, param_max_val, param_name, param_type) in enumerate(
        zip(interest_parameter_values_list, min_val_list, max_val_list, param_names, parameters_type)):
        
        idx = counter % 9
        ax = axes_flat[idx]
        
        # Define the minimum and maximum value if None
        if param_min_val is None:
            param_min_val = min(param_values)
        if param_max_val is None:
            param_max_val = max(param_values)
        
        normalized_values = self.normalize_values(param_values, param_min_val, param_max_val)
        
        # Filter values inside the range
        filtered_values = [value for value in normalized_values if param_min_val <= value <= param_max_val]
        
        # Choose the graphic type based on param_type
        if param_type:
            # Circular graph
            new_ax = self._draw_circular_subplot(ax, filtered_values, param_min_val, param_max_val, 
                                                n_intervals, density, param_name)
            
            # Save y_max for shared scale
            if hasattr(ax, '_y_max'):
                all_y_max.append(ax._y_max)
        else:
            # Linear graph
            new_ax = self._draw_linear_subplot(ax, filtered_values, 0, 1, param_name)
        
        # Update the axes
        axes_flat[idx] = new_ax

        counter = counter + 1

        if counter % 9 == 0 or n_params == counter:
            # Apply the shared scale only for the circular graphs
            if share_scale and all_y_max:
                global_y_max = max(all_y_max)
                for idx, (ax, parameter_type) in enumerate(zip(axes_flat[:n_params], parameters_type)):
                    if parameter_type and hasattr(ax, '_y_max'):
                        ax.set_ylim(0, global_y_max)
            if n_params == counter:
                # Hide the unused subplots
                for idx in range(counter % 9, len(axes_flat)):
                    axes_flat[idx].set_visible(False)
            
            # Title
            fig.suptitle('Multiple Parameter Visualization', fontsize=14, y=1.02)
            fig.savefig(f'image{image_counter}.png', dpi=300, bbox_inches='tight')
            image_counter = image_counter + 1
            
            plt.show()

            if counter < n_params:
                graph_size = min(3, math.ceil(math.sqrt(n_params - counter)))
            
                # Create a default sub-figure for the next iteration
                fig, axes = plt.subplots(graph_size, graph_size, figsize=figsize, squeeze=False)
                
                axes_flat = axes.flatten()

  def _draw_circular_subplot(self, ax, filtered_values, param_min_val, param_max_val, n_intervals, density, param_name):
    """
    Auxiliar function to `multiple_graphics` function.
    
    Draw a circular (polar) subplot for a single parameter.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The original axes (may be cartesian). If it is not already a polar axes,
        it is removed and replaced with a polar axes at the same position.
    filtered_values : array-like
        The data values (already normalised and filtered to the desired range)
        to be plotted.
    param_min_val : float
        Minimum value of the parameter (in original units) used for the plot range.
    param_max_val : float
        Maximum value of the parameter (in original units) used for the plot range.
    n_intervals : int
        Number of bins for the histogram or the number of points for the KDE.
    density : bool
        If True, draw a circular KDE plot; if False, draw a rose diagram
        (circular histogram).
    param_name : str
        Name of the parameter (used for the subplot title).

    Returns
    -------
    matplotlib.axes.Axes
        The polar axes object (either the original `ax` converted or a new one)
        containing the circular plot. The returned axes also has an attribute
        `._y_max` set to the maximum y‑limit (radial limit) of the plot, which
        can be used for sharing scales across subplots.
    """
    
    # Check if the axes is for a circular graph
    if not hasattr(ax, 'set_theta_offset'):
        # Save the axes position
        fig = ax.figure
        pos = ax.get_position()
        margin = 0.1

        new_height = pos.height - margin
        
        new_pos = [pos.x0, pos.y0, pos.width, new_height]
        
        # Remove the old axes
        ax.remove()
        
        # Create a new axes in the same position
        ax = fig.add_axes(new_pos, projection='polar')
    
    # Create bins
    bins = np.linspace(param_min_val, param_max_val, n_intervals + 1)
    
    # Calculate frequencies
    frequencies, _ = np.histogram(filtered_values, bins=bins)
    
    # Create angles for the bars
    angles = np.linspace(0, 2 * np.pi, n_intervals, endpoint=False)
    
    y_max = 0
    
    if density and len(filtered_values) > 1:
        
        kde = gaussian_kde(filtered_values)
        n_smooth_points = 360
        angles_smooth = np.linspace(0, 2 * np.pi, n_smooth_points, endpoint=False)
        values_smooth = np.linspace(param_min_val, param_max_val, n_smooth_points)
        
        density_smooth = kde(values_smooth)
        density_smooth = density_smooth / np.max(density_smooth)
        
        angles_closed = np.append(angles_smooth, angles_smooth[0])
        density_closed = np.append(density_smooth, density_smooth[0])
        
        ax.fill(angles_closed, density_closed, alpha=0.7, color='blue')
        ax.plot(angles_closed, density_closed, color='darkblue', linewidth=2)
        
        y_max = 1.1
    else:
        # Rose diagram
        if len(frequencies) > 0 and max(frequencies) > 0:
            bars = ax.bar(angles, frequencies, width=2*np.pi/n_intervals,
                         align='center', alpha=0.7, edgecolor='white', linewidth=0.5)
            
            # Bar color
            for bar in bars:
                bar.set_facecolor(plt.cm.viridis(0.3))
            
            y_max = max(frequencies) * 1.1
        else:
            ax.text(0, 0, 'No data', ha='center', va='center')
            y_max = 1
        
        # Remove labels from y axis
        ax.set_yticklabels([])
    
    # Save y_max for shared scale
    ax._y_max = y_max
    
    # Angle labels
    label_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    tick_labels = []
    
    for pos_angle in label_angles:
        real_value = param_min_val + (pos_angle / (2 * np.pi)) * (param_max_val - param_min_val)
        real_angle_deg = np.rad2deg(real_value) % 360
        tick_labels.append(f'{real_value:.3f}\n({real_angle_deg:.0f}°)')
    
    ax.set_xticks(label_angles)
    ax.set_xticklabels(tick_labels, fontsize=8)
    
    # Circular graph configurations
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    
    # Title
    ax.set_title(f'{param_name}\nn={len(filtered_values)}', pad=20, fontsize=10)
    
    return ax

  def _draw_linear_subplot(self, ax, filtered_values, param_min_val, param_max_val, param_name, n_intervals = 100):
    """
    Auxiliar function to `multiple_graphics` function.
    
    Draw a linear (cartesian) bar plot subplot for a single parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The original axes (cartesian). It is removed and replaced with a new
        axes positioned with reduced width and height to avoid overlap with
        other plot elements.
    filtered_values : array-like
        The data values (already filtered to the desired range) to be plotted
        as a histogram.
    param_min_val : float
        Minimum value of the parameter (used to set the x‑axis limit).
    param_max_val : float
        Maximum value of the parameter (used to set the x‑axis limit).
    param_name : str
        Name of the parameter (used for the subplot title).
    n_intervals : int, default 20
        Number of bins for the histogram.

    Returns
    -------
    matplotlib.axes.Axes
        The new cartesian axes containing the bar plot. The axes are configured
        with x‑limits from `param_min_val` to `param_max_val`, a grid, and
        labels. The title includes the parameter name and the number of
        filtered observations (`n=...`).
    """

    margin = 0.03
    half_margin = margin / 2

    fig = ax.figure
    pos = ax.get_position()
    
    # Fix the margin between different types of graphs
    new_width = pos.width * (1 - margin)
    new_height = pos.height - margin * 3

    new_left = pos.x0 + half_margin
    
    new_pos = [new_left, pos.y0, new_width, new_height]
    
    # Remove old axes
    ax.remove()
    
    # Create new axes
    ax = fig.add_axes(new_pos)
    
    # Create bins
    bins = np.linspace(param_min_val, param_max_val, n_intervals + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate frequencies
    frequencies, _ = np.histogram(filtered_values, bins=bins)
    
    # Calculate bars weight
    bar_width = (param_max_val - param_min_val) / n_intervals * 0.9
    
    # Create bar plot
    ax.bar(bin_centers, frequencies, width=bar_width,
                 align='center', alpha=0.7, color='skyblue', edgecolor='black')
    
    # Axis configurations
    ax.set_xlim(param_min_val, param_max_val)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Title
    ax.set_title(f'{param_name}\nn={len(filtered_values)}', pad=20, fontsize=10)

    return ax