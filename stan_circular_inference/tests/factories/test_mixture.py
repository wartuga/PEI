import pytest
import random
import numpy as np
from pycircstat2.distributions import vonmises, cardioid, wrapcauchy
from stan_circular_inference.factories.vonmises_factory import VonMisesMK
from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential, Normal
from stan_circular_inference.factories.mixture_factory import Mixture
from stan_circular_inference.service import BayesianInferenceService
from utils import correct_inferred_values
import arviz as az

# Simulation sizes for degradation test
x0 = 10
x1 = 100
x2 = 10000

class MixtureTestBase:
    """
    Base class for testing mixture of circular distributions.
    Subclasses must define the following attributes:
        dist_rvs_1                : function to generate random samples (e.g., vonmises.rvs)
        scale_prior_1             : prior distribution for the scale parameter (e.g., Exponential(0.1))
        scale_name_1              : string identifier for the scale parameter ('kappa' or 'rho')
        model_factory_1           : model class (VonMisesMK or WrappedCauchy)
        real_scale_fixed_1        : fixed scale value for the `circular_mu` and `degradation` tests
        real_scale_random_range_1 : tuple (low, high) for generating random scale in the parameters test
        dist_rvs_2                : function to generate random samples (e.g., vonmises.rvs)
        scale_prior_2             : prior distribution for the scale parameter (e.g., Exponential(0.1))
        scale_name_2              : string identifier for the scale parameter ('kappa' or 'rho')
        model_factory_2           : model class (VonMisesMK or WrappedCauchy)
        real_scale_fixed_2        : fixed scale value for the `circular_mu` and `degradation` tests
        real_scale_random_range_2 : tuple (low, high) for generating random scale in the parameters test
    """

    sample_size = 20         # fewer cycles because mixture is heavier

    def _build_mixture_model(self):
        """Build the mixture model with appropriate priors. Override if needed."""
        # Create component models with their priors
        # (priors are defined in subclass or can be default)
        comp1 = self.model_factory_1(self.mu_prior_1, self.scale_prior_1)
        comp2 = self.model_factory_2(self.mu_prior_2, self.scale_prior_2)
        return Mixture(comp1, comp2)

    def _generate_mixture_data(self, mu1, mu2, scale1, scale2, mixing, size):
        """Generate data from a mixture of two circular distributions."""
        dist1_samples = self.dist_rvs_1(mu=mu1, **{self.scale_name_1: scale1}, size=int(size * mixing))
        dist2_samples = self.dist_rvs_2(mu=mu2, **{self.scale_name_2: scale2}, size=(size - len(dist1_samples)))
        return np.concatenate([dist1_samples, dist2_samples])
    
    def test_circular_mu(self):
        """Fixed parameters: mu1=0, mu2=π, fixed scales and mixing."""
        true_mu1 = 0.0
        true_mu2 = np.pi
        true_scale1 = self.real_scale1_fixed
        true_scale2 = self.real_scale2_fixed
        true_mixing = 0.5

        model = self._build_mixture_model()
        service = BayesianInferenceService(model)

        correct_dist1 = correct_dist2 = correct_mixing = 0
        n_iter = 100

        for _ in range(n_iter):
            samples = self._generate_mixture_data(true_mu1, true_mu2,
                                                  true_scale1, true_scale2,
                                                  true_mixing, self.sample_size)
            data = {'N': self.sample_size, 'values': samples}
            posterior = service.build_model(data)
            fit = service.get_samples(posterior, sample_amount=x2)
            statistics = service.get_statistics(fit, confidence_interval=95)

            mu1_hdi_2_5 = statistics['hdi_2.5%'].mu1
            mu1_hdi_97_5 = statistics['hdi_97.5%'].mu1

            mu2_hdi_2_5 = statistics['hdi_2.5%'].mu2
            mu2_hdi_97_5 = statistics['hdi_97.5%'].mu2

            scale1_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_1}1']
            scale1_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_1}1']

            scale2_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_2}2']
            scale2_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_2}2']

            inferred_dist = [
                0 if sum(1 for v in values if v > 0.5) >= len(values) / 2 else 1
                for values in fit['mixing_weight']
            ]

            real_dist1 = [0] * int(self.sample_size * true_mixing)
            real_dist2 = [1] * int(self.sample_size * (1 - true_mixing))

            real_dist = real_dist1 + real_dist2

            mixture_stats = service.get_mixture_statistics(samples, real_dist, inferred_dist, min_val=0, max_val=2*np.pi)

            if correct_inferred_values(true_mu1, mu1_hdi_2_5, mu1_hdi_97_5, true_scale1, scale1_hdi_2_5, scale1_hdi_97_5):
                correct_dist1 += 1
            if correct_inferred_values(true_mu2, mu2_hdi_2_5, mu2_hdi_97_5, true_scale2, scale2_hdi_2_5, scale2_hdi_97_5):
                correct_dist2 += 1
            if mixture_stats['accuracy'] > 90:
                correct_mixing += 1

        assert correct_dist1 >= 95, f'dist1 coverage: {correct_dist1}%'
        assert correct_dist2 >= 95, f'dist2 coverage: {correct_dist2}%'
        assert correct_mixing >= 95, f'mixture coverage: {correct_mixing}'
        # AssertionError: mixture coverage: 20
        # assert 20 >= 95
    
    def test_parameters_inference(self):
        """Randomly generate true parameters and check 95% HDI coverage over many datasets."""
        # Random true parameters
        true_mu1 = random.uniform(0, 2*np.pi)
        true_mu2 = random.uniform(0, 2*np.pi)
        # Ensure mu1 and mu2 are not too close to avoid label switching ambiguity
        while abs(true_mu1 - true_mu2) < 0.5 and abs(true_mu1 - true_mu2) > 2*np.pi - 0.5:
            true_mu2 = random.uniform(0, 2*np.pi)
        low1, high1 = self.real_scale1_range
        low2, high2 = self.real_scale2_range
        true_scale1 = random.uniform(low1, high1)
        true_scale2 = random.uniform(low2, high2)
        true_mixing = random.uniform(0.2, 0.8)  # avoid extremes

        model = self._build_mixture_model()
        service = BayesianInferenceService(model)

        correct_dist1 = correct_dist2 = correct_mixing = 0
        n_iter = 100

        for _ in range(n_iter):
            # Generate data
            samples = self._generate_mixture_data(true_mu1, true_mu2,
                                                  true_scale1, true_scale2,
                                                  true_mixing, self.sample_size)
            data = {'N': self.sample_size, 'values': samples}

            # Fit
            posterior = service.build_model(data)
            fit = service.get_samples(posterior, sample_amount=x2)  # many samples for accurate HDI

            statistics = service.get_statistics(fit, confidence_interval=95)

            mu1_hdi_2_5 = statistics['hdi_2.5%'].mu1
            mu1_hdi_97_5 = statistics['hdi_97.5%'].mu1

            mu2_hdi_2_5 = statistics['hdi_2.5%'].mu2
            mu2_hdi_97_5 = statistics['hdi_97.5%'].mu2

            scale1_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_1}1']
            scale1_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_1}1']

            scale2_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_2}2']
            scale2_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_2}2']

            if correct_inferred_values(true_mu1, mu1_hdi_2_5, mu1_hdi_97_5, true_scale1, scale1_hdi_2_5, scale1_hdi_97_5):
                correct_dist1 += 1
            if correct_inferred_values(true_mu2, mu2_hdi_2_5, mu2_hdi_97_5, true_scale2, scale2_hdi_2_5, scale2_hdi_97_5):
                correct_dist2 += 1

        assert correct_dist1 >= 95, f"dist1 coverage: {correct_dist1}%"
        assert correct_dist2 >= 95, f"dist2 coverage: {correct_dist2}%"
        

    def test_degradation(self):
        """Check that coverage improves (or does not degrade) as MCMC sample size increases."""
        n_simulations = [x0, x1, x2]
        true_mu1 = 0.0
        true_mu2 = np.pi
        true_scale1 = self.real_scale1_fixed
        true_scale2 = self.real_scale2_fixed
        true_mixing = random.uniform(0.2, 0.8)

        model = self._build_mixture_model()
        service = BayesianInferenceService(model)

        n_cycles = 50 

        # Generate the same set of datasets for all simulation sizes
        datasets = []
        for _ in range(n_cycles):
            samples = self._generate_mixture_data(true_mu1, true_mu2,
                                                  true_scale1, true_scale2,
                                                  true_mixing, self.sample_size)
            datasets.append(samples)

        coverage_10 = coverage_100 = coverage_10000 = None
        mixing_accuracy_10 = mixing_accuracy_100 = mixing_accuracy_10000 = None

        for n_sim in n_simulations:
            correct_dist1 = correct_dist2 = 0
            accuracy = []
            for samples in datasets:
                data = {'N': self.sample_size, 'values': samples}
                posterior = service.build_model(data)
                fit = service.get_samples(posterior, sample_amount=n_sim)
                statistics = service.get_statistics(fit, confidence_interval=95)

                mu1_hdi_2_5 = statistics['hdi_2.5%'].mu1
                mu1_hdi_97_5 = statistics['hdi_97.5%'].mu1

                mu2_hdi_2_5 = statistics['hdi_2.5%'].mu2
                mu2_hdi_97_5 = statistics['hdi_97.5%'].mu2

                scale1_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_1}1']
                scale1_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_1}1']

                scale2_hdi_2_5 = statistics['hdi_2.5%'][f'{self.scale_name_2}2']
                scale2_hdi_97_5 = statistics['hdi_97.5%'][f'{self.scale_name_2}2']

                inferred_dist = [
                    0 if sum(1 for v in values if v > 0.5) >= len(values) / 2 else 1
                    for values in fit['mixing_weight']
                ]

                real_dist1 = [0] * int(self.sample_size * true_mixing)
                real_dist2 = [1] * int(self.sample_size * (1 - true_mixing))

                real_dist = real_dist1 + real_dist2

                mixture_stats = service.get_mixture_statistics(samples, real_dist, inferred_dist, min_val=0, max_val=2*np.pi)

                if correct_inferred_values(true_mu1, mu1_hdi_2_5, mu1_hdi_97_5, true_scale1, scale1_hdi_2_5, scale1_hdi_97_5):
                    correct_dist1 += 1
                if correct_inferred_values(true_mu2, mu2_hdi_2_5, mu2_hdi_97_5, true_scale2, scale2_hdi_2_5, scale2_hdi_97_5):
                    correct_dist2 += 1
                accuracy.append(mixture_stats['accuracy'])

            # fazer o coverage para cada n_sim mas a avaliar com base no número de parâmetros inferidos corretamente
            coverage = (correct_dist1 + correct_dist2) / (n_cycles * 2)
            if n_sim == x0:
                coverage_10 = coverage
                mixing_accuracy_10 = np.mean(accuracy)
            elif n_sim == x1:
                coverage_100 = coverage
                mixing_accuracy_100 = np.mean(accuracy)
            elif n_sim == x2:
                coverage_10000 = coverage
                mixing_accuracy_10000 = np.mean(accuracy)

        assert coverage_10 <= coverage_100 <= coverage_10000
        assert mixing_accuracy_10 <= mixing_accuracy_100 <= mixing_accuracy_10000

class TestMixtureVonMisesVonMises(MixtureTestBase):
    dist_rvs_1 = staticmethod(vonmises.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Exponential(0.1)
    scale_name_1 = 'kappa'
    model_factory_1 = VonMisesMK
    real_scale1_fixed = 2
    real_scale1_range = (0, 7)
    dist_rvs_2 = staticmethod(vonmises.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Exponential(0.1)
    scale_name_2 = 'kappa'
    model_factory_2 = VonMisesMK
    real_scale2_fixed = 5
    real_scale2_range = (0, 7)

class TestMixtureVonMisesCardioid(MixtureTestBase):
    dist_rvs_1 = staticmethod(vonmises.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Exponential(0.1)
    scale_name_1 = 'kappa'
    model_factory_1 = VonMisesMK
    real_scale1_fixed = 2
    real_scale1_range = (0, 7)
    dist_rvs_2 = staticmethod(cardioid.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Normal(0.25, 0.25)
    scale_name_2 = 'rho'
    model_factory_2 = Cardioid
    real_scale2_fixed = 0.3
    real_scale2_range = (0, 0.5)

class TestMixtureVonMisesWrappedCauchy(MixtureTestBase):
    dist_rvs_1 = staticmethod(vonmises.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Exponential(0.1)
    scale_name_1 = 'kappa'
    model_factory_1 = VonMisesMK
    real_scale1_fixed = 2
    real_scale1_range = (0, 7)
    dist_rvs_2 = staticmethod(wrapcauchy.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Normal(0.5, 0.5)
    scale_name_2 = 'rho'
    model_factory_2 = WrappedCauchy
    real_scale2_fixed = 0.7
    real_scale2_range = (0, 1)

class TestMixtureCardioidCardioid(MixtureTestBase):
    dist_rvs_1 = staticmethod(cardioid.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Normal(0.25, 0.25)
    scale_name_1 = 'rho'
    model_factory_1 = Cardioid
    real_scale1_fixed = 0.2
    real_scale1_range = (0, 0.5)
    dist_rvs_2 = staticmethod(cardioid.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Normal(0.25, 0.25)
    scale_name_2 = 'rho'
    model_factory_2 = Cardioid
    real_scale2_fixed = 0.4
    real_scale2_range = (0, 0.5)

class TestMixtureCardioidWrappedCauchy(MixtureTestBase):
    dist_rvs_1 = staticmethod(cardioid.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Normal(0.25, 0.25)
    scale_name_1 = 'rho'
    model_factory_1 = Cardioid
    real_scale1_fixed = 0.3
    real_scale1_range = (0, 0.5)
    dist_rvs_2 = staticmethod(wrapcauchy.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Normal(0.5, 0.5)
    scale_name_2 = 'rho'
    model_factory_2 = WrappedCauchy
    real_scale2_fixed = 0.7
    real_scale2_range = (0, 1)

class TestMixtureWrappedCauchyWrappedCauchy(MixtureTestBase):
    dist_rvs_1 = staticmethod(wrapcauchy.rvs)
    mu_prior_1 = Uniform(0, 2 * np.pi)
    scale_prior_1 = Normal(0.5, 0.5)
    scale_name_1 = 'rho'
    model_factory_1 = WrappedCauchy
    real_scale1_fixed = 0.3
    real_scale1_range = (0, 1)
    dist_rvs_2 = staticmethod(wrapcauchy.rvs)
    mu_prior_2 = Uniform(0, 2 * np.pi)
    scale_prior_2 = Normal(0.5, 0.5)
    scale_name_2 = 'rho'
    model_factory_2 = WrappedCauchy
    real_scale2_fixed = 0.7
    real_scale2_range = (0, 1)

# na secção de testes adicionar um dos submodelos e os resultados dos testes