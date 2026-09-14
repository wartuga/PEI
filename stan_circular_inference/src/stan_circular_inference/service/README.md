# Inference Pipeline

This section covers the provided pipeline, `BayesianInferenceService`, and shows how to create it as well as the available methods. At the end of this chapter, there is an example of its complete usage. For more detailed information about any of the functions, see [bayesian_inference.py](./src/stan_circular_inference/service/bayesian_inference.py)

## Data types

The library supports the following data types:

- **ANGLES** – the range of values is between 0 and 360
- **RADS** – the range of values is between 0 and 2π
- **HOURS** – the range of values is between 0 and 24
- **MONTHS** – the range of values is between 0 and 12
- **PERCENT** – the range of values is between 0 and 1
- **WEEK** – the range of values is between 1 and 7
- **WEEKS** – the range of values is between 1 and 52
- **MONTH** – the range of values is between 1 and 31

The methods normalize the provided values according to the minimum and maximum value of each data type. Although percentage is not considered circular data, it is precisely used for displaying linear plots.

## Stan models

It is mentioned in the user manual that this language does not have rules, but rather recommendations, such as being consistent for better readability or that no line should exceed eighty (80) characters. The following is an example for the user to add their own `foo` function:
```stan
functions{
    real foo(real x, real y){
        return sqrt(x * log(y));
    }
}
```

You can also add the parameters on which to perform the inference, as follows:
```stan
parameters{
    real alpha;
    real beta;
    real<lower=0> sigma;
}
```

There are also transformed parameters, which are useful when one wishes to define variables that depend on other parameters:
```stan
transformed parameters{
    real sigma_inv;
    sigma_inv = 1 / sigma;
}
```

The data to be passed to the model is initialized as follows:
```stan
data{
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
```

The user can also define or compute variables based on the original input data before the model performs sampling, meaning this is executed only once:
```stan
transformed data{
    vector[N] x_centered;
    x_centered = x - mean(x);
}
```

The model is specified in the model block, where the priors are added. An example based on the previous blocks would be:
```stan
model{
    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    sigma ~ cauchy(0, 5);
    // Likelihood
    y ~ normal(alpha * x + beta, sigma);
}
```

## Building Model

The build_model function compiles the Stan model using the provided data dictionary. It relies on the stan.build function from the PyStan package. The function expects a dictionary data whose keys and types correspond exactly to the data variables declared in the Stan model code (e.g., integers, real numbers, arrays). The arguments are given in Table 1. For example, a typical model for circular data might require {'N': 20, 'values': np.array([...])}. The method returns a compiled Stan model object (stan.Model), which can then be used for posterior sampling via the get_samples function. If the compilation fails due to type mismatches or invalid constraints, appropriate exceptions (TypeError, RuntimeError) are raised with actionable suggestions.

## Sampling values

The get_values method extracts posterior samples for specific parameters from a StanFit object (returned by get_samples). The method iterates over the chains stored in fit.stan_outputs, decodes each line (which is in JavaScript Object Notation (JSON) format), and collects the values corresponding to the requested parameter only from lines whose topic is "sample" and that contain a dictionary of values. The result is a dictionary where each key is a parameter name and each value is a list of samples (ordered in the sequence they appear across all chains). To access individual elements of a vector (e.g., theta), use dot notation followed by the index, starting at 1 (e.g., "theta.1", "theta.3"). This method is useful for obtaining raw samples of specific parameters, which can then be used for diagnostics, custom calculations, or visualisations (such as circular_graphic).

## Statistics summary

The `get_statistics` function computes a summary statistics table from a fitted PyStan model, with automatic handling of circular variables. The method first converts the PyStan fit object into an ArviZ `InferenceData` object, then identifies circular variables by selecting all parameter names that start with `mu` (e.g., `mu1` in mixture models). Using ArviZ's `summary` function, it generates default statistics for all parameters, including the mean, standard deviation, effective sample size, and $\hat{R}$, while also specifying the circular variables and the confidence interval level for the Highest Density Interval (HDI), given by the user or its default value of 89, due to a reference of [@Hackers].

For circular variables, the method adjusts the mean and HDI bounds by wrapping them modulo $2\pi$ to ensure that the values remain within the interval $[0, 2\pi)$. The resulting summary is printed and returned as a pandas DataFrame. This method is useful for obtaining a concise and interpretable summary of posterior distributions.

## Visualization

To help users visualise inference results, several methods are available for different use cases. All circular plots share the same polar configuration: $0^\circ$ at the top, clockwise direction, and tick labels every $45^\circ$ showing both the raw value and the corresponding angle in degrees. Data are normalised to the interval $[0, 2\pi)$ using `normalize_values` based on the given `data_type` (radians or degrees). The visible angular range can be restricted by `min_val` and `max_val`; if omitted, the full data range is used.

The `circular_graphic` method generates a single circular (polar) histogram or a smooth density plot for a set of angular data.

- **Histogram mode** (`density=False`): draws a rose diagram (bars/wedges) with frequencies, coloured by the Viridis colormap. The radial limit is set to $1.1 \times \max(\text{frequencies})$.
- **Density mode** (`density=True`): fits a Gaussian KDE, normalises the curve to a peak height of 1, and fills the area under the curve. The radial axis is fixed to $[0, 1.1]$.

The plot title is "Diagrama de rosas". The method calls `plt.show()` and returns nothing.

![Rose Diagram](stan_circular_inference\src\images\rose_diagram.png)
![Density Circular Graph](stan_circular_inference\src\images\circular_graphic_density.png)

The `match_points_to_distributions` function overlays ground‑truth and inferred cluster assignments on a circular histogram. Designed for evaluating mixture models or binary classification where each point has a true label (0 or 1) and a predicted label. The figure combines three layers:

- A rose diagram (bars) showing the overall angular distribution.
- Unfilled squares representing the true attribution (edge colours from `colors`).
- Tight plus signs representing the inferred attribution (filled with the corresponding colour).

Markers are placed at a fixed radius just above the highest bar. The method also computes and displays overall accuracy, confusion matrix, per‑class accuracies, and point counts. It returns a dictionary with these statistics.

![Match Points to Distribution Graph](stan_circular_inference\src\images\match_points_to_distribution_mixture.png)

The `multiple_graphics` method generates a grid of plots for several parameters simultaneously. It supports both circular (polar) and linear visualisations, determined by the `parameters_type` list. This method is particularly useful for inspecting posterior distributions of multiple variables (e.g., mixture components) in a compact, organised layout. The function automatically arranges the plots in up to $3 \times 3$ subplots per figure, saving each full figure as a PNG file (named `image1.png`, `image2.png`, …) and displaying it interactively by calling `plt.show()`. To avoid file overwriting, the user must save the PNG files to a different location before re‑executing the program that calls this method, as the method will overwrite the files with the most recent results.

![Multiple Graphs](stan_circular_inference\src\images\multiple_graphs.png)

## Mixture models methods

One method for mixture models, `match_points_to_distributions`, has already been mentioned. Additionally, `get_mixture_statistics` computes the overall accuracy, per‑distribution accuracy, and confusion matrix, while the helper `show_mixture_statistics` pretty‑prints the output of `get_mixture_statistics`.

The `get_mixture_statistics` function computes classification accuracy and confusion matrix statistics from the true and inferred labels. After validating that all input arrays have equal length, it normalises the values to the target range (e.g., $[0, 2\pi)$ for `DataType.RADS` or $[0, 1]$ for `DataType.PERCENT`) and filters the observations that fall within `[min_val, max_val]`. The overall accuracy is defined as the percentage of correctly labelled observations among the filtered set. The per‑class accuracies are:

- Accuracy for class 0 (distribution 1):  
  $\displaystyle \frac{TP}{TP+FP} \times 100$,  
  where $TP$ = true positives (real 0 and inferred 0), $FP$ = false positives (real 0 but inferred 1).

- Accuracy for class 1 (distribution 2):  
  $\displaystyle \frac{TN}{TN+FN} \times 100$,  
  where $TN$ = true negatives (real 1 and inferred 1), $FN$ = false negatives (real 1 but inferred 0).

The method returns a dictionary containing the overall accuracy, per‑class accuracies, and the confusion matrix counts ($TP$, $FN$, $FP$, $TN$).

## Missing data
In real-world data, it is common to encounter missing data, and there are various formats depending on how different individuals fill them out, such as Null, None, -1, -99999, etc. This typically depends on the domain of the sample. To simplify the logic of the methods, the user should remove these missing data points. However, this process results in loss of information, which may or may not be valid, thus reducing the accuracy of the presented result.

## Example
An example of using the tool on the dataset [@wind_dataset] is found where wind measurements are taken at five different positions. The interest in this case is the wind direction, and therefore the focus is on the values found in the column wind_dir.

```python
import pandas as pd

df = pd.read_csv('src/datasets/wind_dataset.csv')

# Priors
mu = Uniform(0, 2*np.pi)
kappa = Exponential(0.1)

# Stan model of Von Mises distribution
model = VonMisesMK(mu, kappa)

# Gets the values in the wind_dir column
values = np.array(df['wind_dir'], dtype='float')

model_data = {'N': len(values), 'values': values}

service = BayesianInferenceService(model)

interest_parameter_values = service.get_pystan_statistics(data=model_data,  parameters=['mu', 'kappa'], sample_amount=10000)

# Visualize the inferred values for mu parameter
service.circular_graphic(interest_parameter_values['mu'], min_val=0, max_val=2*np.pi)
```
