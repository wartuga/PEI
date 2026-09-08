# all the dependencies needed
from pycircstat2.distributions import wrapcauchy, cardioid
import numpy as np
from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy
from stan_circular_inference.factories.mixture_factory import Mixture
from stan_circular_inference.factories.distributions_factory import Normal, Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

# setup to generate samples from distributions to test the model via pycircstat2
size = 50
mu1, rho1 = np.pi / 4, 0.2
mu2, rho2 = 7/4 * np.pi, 0.7
samples1 = cardioid.rvs(size=size, mu=mu1, rho=rho1)
samples2 = wrapcauchy.rvs(size=size, mu=mu2, rho=rho2)

# priors for distribution 1
mu1 = Uniform(0, 2*np.pi)
rho1 = Normal(0.25, 0.25)

# priors for distribution 2
mu2 = Uniform(0, 2*np.pi)
rho2 = Normal(0.5, 0.5)

# distribution 1 is a Cardioid
dist1 = Cardioid(mu1, rho1)
# distribution 2 is a wrapped Cauchy
dist2 = WrappedCauchy(mu2, rho2)

# creates the mixture model of distribution 1 and distribution 2
model = Mixture(dist1, dist2)

# initializes the service with the mixture model
service = BayesianInferenceService(model)

# merge the samples from two different distributions
samples = np.concatenate([samples1, samples2])

# inicialize the model data
data = {
    'N': len(samples),
    'values': samples
}

# generated data visualization
service.circular_graphic(samples, min_val=0, max_val=2*np.pi)

# builds the model
posterior = service.build_model(data)
# does 500 simulations in each of the 4 chains
fit = service.get_samples(posterior, size*10)
# get the values for all the passed parameters (mean value, scale parameter and one point of each distribution)
values = service.get_values(fit, parameters=['mu1', 'rho1', 'mu2', 'rho2', 'mixing_weight.1', 'mixing_weight.51'])

# get the model statistics as a dataframe
service.get_statistics(fit)

# plot all the parameters passed
service.multiple_graphics(values, param_names=['mu1', 'rho1', 'mu2', 'rho2', 'mixing_weight.1', 'mixing_weight.51'], min_val=0, max_val=2*np.pi, parameters_type=[True, False, True, False, False, False])