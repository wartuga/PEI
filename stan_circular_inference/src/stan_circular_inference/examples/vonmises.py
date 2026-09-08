from pycircstat2.distributions import vonmises
import matplotlib.pyplot as plt
import numpy as np

size = 50
mu, kappa = np.pi/4, 5
x = vonmises.rvs(size=size, mu=mu, kappa=kappa)

from stan_circular_inference.factories.vonmises_factory import VonMisesMK
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

mu = Uniform(0, 2*np.pi)
kappa = Exponential(0.1)

model1 = VonMisesMK(mu, kappa)
service = BayesianInferenceService(model1)

data = {
    'N': len(x),
    'values': x
}

service.circular_graphic(x, min_val=0, max_val=2*np.pi)

# aumentar # de samples
posterior = service.build_model(data)
fit = service.get_samples(posterior, size*1000)
values = service.get_values(fit, parameters=['mu', 'kappa'])

service.get_statistics(fit)

mean_values = np.mean(x)
mean_samples_mu = np.mean(values['mu'])
mean_samples_kappa = np.mean(values['kappa'])

print(mean_values)
print(mean_samples_mu)
print(mean_samples_kappa)

service.circular_graphic(values['mu'], min_val=0, max_val=2*np.pi)

plt.hist(values['kappa'], bins=30, color='steelblue', edgecolor='black')
plt.xlim(0, 10)
plt.xlabel('Valor de kappa')
plt.ylabel('Frequência absoluta')
plt.title('Histograma de kappa')
plt.show()