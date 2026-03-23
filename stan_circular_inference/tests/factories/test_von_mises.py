import pytest
from pycircstat2.distributions import vonmises as vonmises
import random
import numpy as np
from stan_circular_inference.factories.vonmises_factory import VonMisesMK
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential
from stan_circular_inference.service import BayesianInferenceService

class VonMisesTest:
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

            fit = service.get_samples(posterior, sample_amount=sample_size * 1000)

            statistics = service.get_statistics(fit, confidence_interval=95)
            
            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
            kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

            if (real_mu > mu_hdi_2_5 and real_mu < mu_hdi_97_5 or real_mu > mu_hdi_97_5 and real_mu < (mu_hdi_2_5 + (2*np.pi)) or real_mu < mu_hdi_2_5 and real_mu > (mu_hdi_97_5 - (2*np.pi))) and real_kappa > kappa_hdi_2_5 and real_mu < kappa_hdi_97_5:
                correct = correct + 1
        
        assert correct >= 95

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

            fit = service.get_samples(posterior, sample_amount=sample_size * 100)

            statistics = service.get_statistics(fit, confidence_interval=95)

            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu

            kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
            kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

            if (real_mu > mu_hdi_2_5 and real_mu < mu_hdi_97_5 or real_mu > mu_hdi_97_5 and real_mu < (mu_hdi_2_5 + (2*np.pi)) or real_mu < mu_hdi_2_5 and real_mu > (mu_hdi_97_5 - (2*np.pi))) and real_kappa > kappa_hdi_2_5 and real_mu < kappa_hdi_97_5:
                correct = correct + 1
        
        assert correct >= 95

    @pytest.mark.parametrize("n_simulations", [
        100,
        1000,
        10000,
        100000
    ])
    def test_degradation(self):
        '''
        iter_1:
        coverage n_simulations=100: 0.98
        coverage n_simulations=1000: 1.0
        coverage n_simulations=10000: 0.97
        coverage n_simulations=100000: 0.95

        iter_2:
        coverage n_simulations=100: 0.99
        coverage n_simulations=1000: 0.98
        coverage n_simulations=10000: 0.98
        coverage n_simulations=100000: 0.97

        iter_3:
        coverage n_simulations=100: 0.98
        coverage n_simulations=1000: 0.99
        coverage n_simulations=10000: 0.91
        coverage n_simulations=100000: 0.91
        '''
        n_simulations = [100,
        1000,
        10000,
        100000]

        sample_size = 20
        real_mu = 0
        real_kappa = 2
        mu = Uniform(0, 2*np.pi)
        kappa = Exponential(0.1)
        model = VonMisesMK(mu, kappa)
        service = BayesianInferenceService(model)

        coverage = ''

        for n_sim in n_simulations:

            correct = 0
            n_cycles = 100

            for _ in range(n_cycles):
                samples = vonmises.rvs(kappa=real_kappa, mu=real_mu, size=sample_size)

                data = {
                    'N' : sample_size,
                    'values' : samples
                }

                posterior = service.build_model(data)

                fit = service.get_samples(posterior, sample_amount=n_sim)

                statistics = service.get_statistics(fit, confidence_interval=95)

                mu_hdi_2_5 = statistics['hdi_2.5%'].mu
                mu_hdi_97_5 = statistics['hdi_97.5%'].mu

                kappa_hdi_2_5 = statistics['hdi_2.5%'].kappa
                kappa_hdi_97_5 = statistics['hdi_97.5%'].kappa

                if (real_mu > mu_hdi_2_5 and real_mu < mu_hdi_97_5 or real_mu > mu_hdi_97_5 and real_mu < (mu_hdi_2_5 + (2*np.pi)) or real_mu < mu_hdi_2_5 and real_mu > (mu_hdi_97_5 - (2*np.pi))) and real_kappa > kappa_hdi_2_5 and real_mu < kappa_hdi_97_5:
                    correct = correct + 1
            
            coverage = coverage + (f'coverage {n_sim}: {correct/n_cycles}\n')
        
        print(coverage)
        
    
test_class = VonMisesTest() # valores obtidos no último teste
#test_class.test_confidence_95() # 98
#test_class.test_circular_mu() # 97
test_class.test_degradation()