# Probabilistic Models

The library is built around a small set of **circular distributions** and a common abstract interface that lets you plug in your own models. Every distribution provided out of the box is characterised by:

- a **location parameter** `μ ∈ [0, 2π)` — the mean direction (the mode);
- a **concentration parameter** (`κ` or `ρ`, always non-negative) — controlling how tightly the data clusters around `μ`.

All of them reduce to the **circular uniform** distribution in the limit of zero concentration, and all are **reflectively symmetric** about `μ`.

Inference follows the Bayesian paradigm. Given observed data `D`, we target the posterior

```
p(θ | D) ∝ p(D | θ) · p(θ)
```

where `p(D | θ)` is the likelihood, `p(θ)` the prior, and `p(θ | D)` the posterior. The evidence `p(D)` is dropped since it does not depend on `θ`. Each model below specifies its own likelihood and default priors.

---

## `ProbabilisticModel` — abstract base class

All models implement the abstract class `ProbabilisticModel`. Subclassing it is the way to add new distributions. Two methods must be implemented:

```python
class ProbabilisticModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}

    @abstractmethod
    def gen_stan_model(self) -> str:
        """Return the Stan code of the model."""
        ...

    @abstractmethod
    def get_parameters_prior(self) -> Dict[str, Any]:
        """Return the parameter priors."""
        ...
```

You have three ways to define a model:

1. Write the Stan code in a `.txt` file and load it with Python.
2. Define the Stan code inline as a Python string.
3. Use one of the built-in models: `VonMisesMK`, `VonMisesM`, `Cardioid`, `WrappedCauchy`, `Mixture`.

---

## `VonMisesMK` — von Mises (unknown κ)

The von Mises is the circular analogue of the normal distribution. Density:

```
f(θ | μ, κ) = exp(κ · cos(θ − μ)) / (2π · I₀(κ))
```

where `I₀(κ)` is the modified Bessel function of the first kind of order 0, acting as the normalising constant. As `κ → 0` the distribution becomes uniform; as `κ → ∞` it concentrates at `μ`.

**Use when:** you want to infer **both** `μ` and `κ` from the data.

```python
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.factories.von_mises_factory import VonMisesMK

mu    = Uniform(0, 2 * np.pi)
kappa = Exponential(0.1)

model = VonMisesMK(mu, kappa)
```

---

## `VonMisesM` — von Mises (known κ)

Same distribution, but `κ` is **fixed by the user** and only `μ` is inferred. This avoids the "zero trick" used in BUGS (not available in Stan) and is computationally cheaper.

**Use when:** the concentration is known a priori or estimated separately.

```python
model = VonMisesM(mu=Uniform(0, 2 * np.pi))
```

`κ` is then supplied as part of the data dictionary.

---

## `Cardioid`

A simpler circular distribution with bounded support for `ρ`:

```
f(θ | μ, ρ) = (1 / 2π) · (1 + 2ρ · cos(θ − μ)),    0 ≤ ρ < 0.5
```

`μ` is the mean direction, `ρ` controls concentration. `ρ = 0` gives the uniform. The constraint `ρ < 0.5` ensures the density stays non-negative. Suited to unimodal symmetric data with **moderate dispersion**.

For Stan, the log-density is used:

```
log f(θ | μ, ρ) = −log(2π) + log(1 + 2ρ · cos(θ − μ))
```

```python
from stan_circular_inference.factories.cardioid_factory import Cardioid

mu  = Uniform(0, 2 * np.pi)
rho = Normal(0.25, 0.25)

model = Cardioid(mu, rho)
```

---

## `WrappedCauchy`

Obtained by wrapping a Cauchy distribution around the unit circle:

```
f(θ | μ, ρ) = (1 / 2π) · (1 − ρ²) / (1 + ρ² − 2ρ · cos(θ − μ)),    0 ≤ ρ < 1
```

Heavier tails than the von Mises → **more robust to outliers**. `ρ → 0` gives the uniform, `ρ → 1` concentrates at `μ`.

Log-density used in Stan:

```
log f(θ | μ, ρ) = −log(2π) + log(1 − ρ²) − log(1 + ρ² − 2ρ · cos(θ − μ))
```

```python
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy

mu  = Uniform(0, 2 * np.pi)
rho = Normal(0.5, 0.5)

model = WrappedCauchy(mu, rho)
```

---

## Priors

Priors are hyper-parameters over the model parameters. The library ships with:

| Prior | Implements | Typical use |
|---|---|---|
| `Normal` | `MeanParameter` | location priors |
| `Uniform` | `MeanParameter` | location priors |
| `Bernoulli` | `MeanParameter` | binary parameters |
| `Poisson` | `MeanParameter` | count parameters |
| `Gamma` | `VarianceParameter` | concentration priors |
| `Exponential` | `VarianceParameter` | concentration priors |

They are passed directly to the model constructor and translated into Stan via `get_code()`.

---

## `Mixture` — mixture of two circular distributions

`Mixture` combines **two** `ProbabilisticModel` instances. Any pairing of `VonMisesMK`, `VonMisesM`, `Cardioid`, or `WrappedCauchy` is supported.

```python
from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy
from stan_circular_inference.factories.mixture_factory import Mixture
from stan_circular_inference.factories.distributions_factory import Normal, Uniform

# Component 1 — Cardioid
mu1  = Uniform(0, 2 * np.pi)
rho1 = Normal(0.25, 0.25)
dist1 = Cardioid(mu1, rho1)

# Component 2 — Wrapped Cauchy
mu2  = Uniform(0, 2 * np.pi)
rho2 = Normal(0.5, 0.5)
dist2 = WrappedCauchy(mu2, rho2)

model = Mixture(dist1, dist2)
```

The mixture likelihood is evaluated on the log scale using `log_sum_exp` to prevent under/overflow:

```
target += log_sum_exp(
    log(w[n])    + log p₁(y[n] | μ₁, ρ₁),
    log(1−w[n]) + log p₂(y[n] | μ₂, ρ₂)
)
```

The model infers the parameters of both components **and** the per-point mixing weights `w[n] ∈ [0, 1]`. A point belongs to component 1 when its mean mixing weight is `< 0.5`.

> **Stability note:** in mixture models the von Mises concentration is bounded to `0 ≤ κ ≤ 50`. In single-distribution models instability typically appears only around `κ ≥ 100`, but mixture models were observed to diverge earlier.

> **Reserved words:** to support the Cardioid and Wrapped Cauchy distributions, the keywords `cardioid`, `cardioid_lpdf`, `wrapped_cauchy`, and `wrapped_cauchy_lpdf` become reserved within the library — the inference service injects their Stan implementations transparently.

---

## Summary table

| Class | Distribution | Infers | Constraint |
|---|---|---|---|
| `VonMisesMK` | von Mises | `μ`, `κ` | `κ ≥ 0` |
| `VonMisesM` | von Mises | `μ` (κ fixed) | user supplies `κ` |
| `Cardioid` | Cardioid | `μ`, `ρ` | `0 ≤ ρ < 0.5` |
| `WrappedCauchy` | Wrapped Cauchy | `μ`, `ρ` | `0 ≤ ρ < 1` |
| `Mixture` | mixture of two | both components + weights | see per-component |

---

## Label switching (mixture models)

Mixture inference is affected by the well-known **label switching** problem: the two components can swap identities between chains, which can make accuracy metrics look near 0% even when the model is correct. The library deliberately does **not** apply an automatic fix, since there is no generally satisfying solution (see Thompson, 2014); the user decides how to handle it.