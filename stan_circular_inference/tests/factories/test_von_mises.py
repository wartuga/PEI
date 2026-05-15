import pytest
from pycircstat2.distributions import vonmises as vonmises
import random
import numpy as np
from stan_circular_inference.factories.vonmises_factory import VonMisesMK
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService
from utils import correct_inferred_values

# simulações com x(n+1) = x(n)^2, x=10 && n=0,1,2...
x0 = 10
x1 = 100
x2 = 10000

class TestVonMises:
    def test_parameters_inference(self):
        sample_size = 20
        real_mu = random.uniform(0, 2*np.pi)
        print(f'real mu {real_mu}')
        real_kappa = random.uniform(0, 7)  # não funciona com np.inf
        print(f'real kappa {real_kappa}')

        mu = Uniform(0, 2*np.pi)
        kappa = Exponential(0.1)
        model = VonMisesMK(mu, kappa)
        service = BayesianInferenceService(model)

        correct = 0

        for _ in range(100):
            samples = vonmises.rvs(kappa=real_kappa, mu=real_mu, size=sample_size)

            data = {
                'N' : sample_size,
                'values' : samples
            }

            posterior = service.build_model(data)

            fit = service.get_samples(posterior, sample_amount=x2)

            statistics = service.get_statistics(fit, confidence_interval=95)
            
            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
            kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_kappa, kappa_hdi_2_5, kappa_hdi_97_5):
                correct = correct + 1
        
        assert correct >= 95 # recolher os valores obtidos e adicionar ao relatório produzir a média e o desvio padrão em x execuções

    def test_circular_mu(self):
        sample_size = 20
        real_mu = 0
        real_kappa = 2
        mu = Uniform(0, 2*np.pi)
        kappa = Exponential(0.1)
        model = VonMisesMK(mu, kappa)
        service = BayesianInferenceService(model)

        correct = 0

        for _ in range(100):
            samples = vonmises.rvs(kappa=real_kappa, mu=real_mu, size=sample_size)

            data = {
                'N' : sample_size,
                'values' : samples
            }

            posterior = service.build_model(data)

            fit = service.get_samples(posterior, sample_amount=x2)

            statistics = service.get_statistics(fit, confidence_interval=95)

            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
            kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_kappa, kappa_hdi_2_5, kappa_hdi_97_5):
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
        real_kappa = 2
        mu = Uniform(0, 2*np.pi)
        kappa = Exponential(0.1)
        model = VonMisesMK(mu, kappa)
        service = BayesianInferenceService(model)

        samples = []

        n_cycles = 100

        for _ in range(n_cycles):
            samples.append(vonmises.rvs(kappa=real_kappa, mu=real_mu, size=sample_size))

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

                kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
                kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

                if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5, real_kappa, kappa_hdi_2_5, kappa_hdi_97_5):
                    correct = correct + 1

            if n_sim == 10:
                coverage_10 = correct / n_cycles
            if n_sim == 100:
                coverage_100 = correct / n_cycles
            if n_sim == 10000:
                coverage_10000 = correct / n_cycles
        
        assert coverage_10 <= coverage_100 <= coverage_10000
    
#test_class = TestVonMises() # valores obtidos no último teste
#test_class.test_confidence_95() # 98
#test_class.test_circular_mu() # 97
#test_class.test_degradation()