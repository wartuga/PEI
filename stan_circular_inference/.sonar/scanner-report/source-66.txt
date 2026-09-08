import pytest
from pycircstat2.distributions import cardioid as cardioid
import random
import numpy as np
from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.distributions_factory import Uniform, Normal
from stan_circular_inference.service import BayesianInferenceService
from utils import correct_inferred_values

# simulações com x(n+1) = x(n)^2, x=10 && n=0,1,2...
x0 = 10
x1 = 100
x2 = 10000

class TestCardioid:
    def test_parameters_inference(self):
        sample_size = 20
        real_mu = random.uniform(0, 2*np.pi)
        print(f'real mu {real_mu}')
        real_rho = random.uniform(0, 0.5)
        print(f'real rho {real_rho}')

        mu = Uniform(0, 2*np.pi)
        rho = Normal(0.25, 0.25)
        model = Cardioid(mu, rho)
        service = BayesianInferenceService(model)

        correct = 0

        for _ in range(100):
            samples = cardioid.rvs(rho=real_rho, mu=real_mu, size=sample_size)

            data = {
                'N' : sample_size,
                'values' : samples
            }

            posterior = service.build_model(data)

            fit = service.get_samples(posterior, sample_amount=x2)

            statistics = service.get_statistics(fit, confidence_interval=95)
            
            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            rho_hdi_2_5 = statistics['hdi_2.5%'].rho
            rho_hdi_97_5 = statistics['hdi_97.5%'].rho

            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_rho, rho_hdi_2_5, rho_hdi_97_5):
                correct = correct + 1
        
        assert correct >= 95

    def test_circular_mu(self):
        sample_size = 20
        real_mu = 0
        real_rho = 0.3
        mu = Uniform(0, 2*np.pi)
        rho = Normal(0.25, 0.25)
        model = Cardioid(mu, rho)
        service = BayesianInferenceService(model)

        correct = 0

        for _ in range(100):
            samples = cardioid.rvs(rho=real_rho, mu=real_mu, size=sample_size)

            data = {
                'N' : sample_size,
                'values' : samples
            }

            posterior = service.build_model(data)

            fit = service.get_samples(posterior, sample_amount=x2)

            statistics = service.get_statistics(fit, confidence_interval=95)

            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            rho_hdi_2_5 = statistics['hdi_2.5%'].rho
            rho_hdi_97_5 = statistics['hdi_97.5%'].rho

            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_rho, rho_hdi_2_5, rho_hdi_97_5):
                correct = correct + 1
        
        assert correct >= 95

    def test_degradation(self):
        
        n_simulations = [
            x0,
            x1,
            x2
        ]

        sample_size = 20
        real_mu = 0
        real_rho = 0.3
        mu = Uniform(0, 2*np.pi)
        rho = Normal(0.25, 0.25)
        model = Cardioid(mu, rho)
        service = BayesianInferenceService(model)

        samples = []

        n_cycles = 100

        for _ in range(n_cycles):
            samples.append(cardioid.rvs(rho=real_rho, mu=real_mu, size=sample_size))

        coverage_10 = 0
        coverage_100 = 0
        coverage_10000 = 0

        for n_sim in n_simulations:

            correct = 0

            for sample in samples:
                
                data = {
                    'N' : sample_size,
                    'values' : sample
                }

                posterior = service.build_model(data)

                fit = service.get_samples(posterior, sample_amount=n_sim)

                statistics = service.get_statistics(fit, confidence_interval=95)

                mu_hdi_2_5 = statistics['hdi_2.5%'].mu
                mu_hdi_97_5 = statistics['hdi_97.5%'].mu

                rho_hdi_2_5 = statistics['hdi_2.5%'].rho
                rho_hdi_97_5 = statistics['hdi_97.5%'].rho

                if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_rho, rho_hdi_2_5, rho_hdi_97_5):
                    correct = correct + 1

            if n_sim == 10:
                coverage_10 = correct / n_cycles
            if n_sim == 100:
                coverage_100 = correct / n_cycles
            if n_sim == 10000:
                coverage_10000 = correct / n_cycles
        
        assert coverage_10 <= coverage_100 <= coverage_10000 # equals since it can fail in case: 1.0 < 1.0

test_class = TestCardioid()
test_class.test_circular_mu()