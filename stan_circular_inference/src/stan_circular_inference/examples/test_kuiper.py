from pycircstat2.distributions import cardioid
import numpy as np
from scipy import stats
from astropy.stats import kuiper_two

size = 5000
real_mu, real_rho = np.pi/4, 0.3
x = cardioid.rvs(size=size, mu=real_mu, rho=real_rho)

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

# service.circular_graphic(x, min_val=0, max_val=2*np.pi)

# aumentar # de samples
posterior = service.build_model(data)
fit = service.get_samples(posterior, size*6)
values = service.get_values(fit, parameters=['mu', 'rho'])

ks_mu_stat, mu_p_value = kuiper_two([real_mu], values['mu'])
ks_rho_stat, rho_p_value = kuiper_two([real_rho], values['rho'])

alpha = 0.05
correct_mu = correct_rho = 0

if mu_p_value < alpha:
    print('correct mu')
    correct_mu += 1

if rho_p_value < alpha:
    print('correct rho')
    correct_rho

print(mu_p_value)
print(ks_mu_stat)

print(rho_p_value)
print(ks_rho_stat)

'''
correct mu
correct rho
(-9.394049808908456e-05+2.618217995763186e-05j)
1.0
(-9.394049808908456e-05+2.618217995763186e-05j)
1.0
'''