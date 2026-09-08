from pycircstat2.distributions import cardioid
import numpy as np

size = 50
mu, rho = np.pi/4, 0.3
x = cardioid.rvs(size=size, mu=mu, rho=rho)

from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.distributions_factory import Normal, Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

mu = Uniform(0, 2*np.pi)
rho = Normal(0.25, 0.25)

model1 = Cardioid(mu, rho)
service = BayesianInferenceService(model1)

data = {
    'N': len(x),
    'values': x
}

service.circular_graphic(x, min_val=0, max_val=2*np.pi)

# aumentar # de samples
posterior = service.build_model(data)
fit = service.get_samples(posterior, size*1000)
values = service.get_values(fit, parameters=['mu', 'rho'])

service.get_statistics(fit)

service.multiple_graphics(values, param_names=['mu', 'rho'], min_val=0, max_val=2*np.pi, parameters_type=[True, False])

mean_values = np.mean(x)
mean_samples_mu = np.mean(values['mu'])
mean_samples_rho = np.mean(values['rho'])

print(mean_values)
print(mean_samples_mu)
print(mean_samples_rho)