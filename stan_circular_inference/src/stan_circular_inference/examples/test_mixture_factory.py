from pycircstat2.distributions import wrapcauchy, cardioid
import matplotlib.pyplot as plt
import numpy as np

size = 50
mu1, rho1 = np.pi / 4, 0.2 
mu2, rho2 = 7/4 * np.pi, 0.7
samples1 = cardioid.rvs(size=size, mu=mu1, rho=rho1)
samples2 = wrapcauchy.rvs(size=size, mu=mu2, rho=rho2)

from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy
from stan_circular_inference.factories.mixture_factory import Mixture
from stan_circular_inference.factories.distributions_factory import Normal, Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

mu1 = Uniform(0, 2*np.pi)
rho1 = Normal(0.25, 0.25)

mu2 = Uniform(0, 2*np.pi)
rho2 = Normal(0.5, 0.5)

dist1 = Cardioid(mu1, rho1)
dist2 = WrappedCauchy(mu2, rho2)

model = Mixture(dist1, dist2)

service = BayesianInferenceService(model)

samples = np.concatenate([samples1, samples2])

print(samples)

data = {
    'N': len(samples),
    'values': samples
}

service.circular_graphic(samples, min_val=0, max_val=2*np.pi)

# aumentar # de samples
posterior = service.build_model(data)
fit = service.get_samples(posterior, size*10)
values = service.get_values(fit, parameters=['mu1', 'rho1', 'mu2', 'rho2', 'mixing_weight.1', 'mixing_weight.51'])

service.get_statistics(fit)

service.multiple_graphics(values, param_names=['mu1', 'rho1', 'mu2', 'rho2', 'mixing_weight.1', 'mixing_weight.51'], min_val=0, max_val=2*np.pi, parameters_type=[True, False, True, False, False, False])