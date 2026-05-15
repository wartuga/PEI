import pytest
import random
import numpy as np
from pycircstat2.distributions import vonmises, cardioid, wrapcauchy
from stan_circular_inference.factories.vonmises_factory import VonMisesMK
from stan_circular_inference.factories.cardioid_factory import Cardioid
from stan_circular_inference.factories.wrapped_cauchy_factory import WrappedCauchy
from stan_circular_inference.factories.distributions_factory import Uniform, Exponential, Normal
from stan_circular_inference.service import BayesianInferenceService
from utils import correct_inferred_values

# Simulation sizes for degradation test
x0 = 10
x1 = 100
x2 = 10000

class BaseCircularTest:
    """
    Base class for circular distribution tests.
    Subclasses must define the following attributes:
        dist_rvs            : function to generate random samples (e.g., vonmises.rvs)
        scale_prior         : prior distribution for the scale parameter (e.g., Exponential(0.1))
        scale_name          : string identifier for the scale parameter ('kappa' or 'rho')
        model_factory       : model class (VonMisesMK or WrappedCauchy)
        real_scale_fixed    : fixed scale value for the `circular_mu` and `degradation` tests
        real_scale_random_range : tuple (low, high) for generating random scale in the parameters test
    """
    # Default test settings (can be overridden by subclasses)
    sample_size = 20
    n_cycles = 100       # number of datasets for degradation test

    def test_parameters_inference(self):
        """Check that 95% HDI coverage is at least 95% for randomly chosen parameters."""
        real_mu = random.uniform(0, 2 * np.pi)
        low, high = self.real_scale_random_range
        real_scale = random.uniform(low, high)
        print(f'real mu {real_mu}')
        print(f'real {self.scale_name} {real_scale}')

        mu_prior = Uniform(0, 2 * np.pi)
        model = self.model_factory(mu_prior, self.scale_prior)
        service = BayesianInferenceService(model)

        correct = 0
        for _ in range(100):
            # Generate data from the true distribution
            samples = self.dist_rvs(**{self.scale_name: real_scale,
                                       'mu': real_mu,
                                       'size': self.sample_size})
            data = {'N': self.sample_size, 'values': samples}

            # Fit the model
            posterior = service.build_model(data)
            fit = service.get_samples(posterior, sample_amount=x2)
            statistics = service.get_statistics(fit, confidence_interval=95)

            # Extract HDIs
            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu
            scale_hdi_2_5 = statistics['hdi_2.5%'][self.scale_name]
            scale_hdi_97_5 = statistics['hdi_97.5%'][self.scale_name]

            # Check coverage
            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5,
                                       real_scale, scale_hdi_2_5, scale_hdi_97_5):
                correct += 1

        assert correct >= 95, f"Coverage for {self.scale_name} was {correct}%, expected >=95%"

    def test_circular_mu(self):
        """Check coverage when mu is fixed at 0 and scale is fixed."""
        real_mu = 0
        real_scale = self.real_scale_fixed

        mu_prior = Uniform(0, 2 * np.pi)
        model = self.model_factory(mu_prior, self.scale_prior)
        service = BayesianInferenceService(model)

        correct = 0
        for _ in range(100):
            samples = self.dist_rvs(**{self.scale_name: real_scale,
                                       'mu': real_mu,
                                       'size': self.sample_size})
            data = {'N': self.sample_size, 'values': samples}

            posterior = service.build_model(data)
            fit = service.get_samples(posterior, sample_amount=x2)
            statistics = service.get_statistics(fit, confidence_interval=95)

            mu_hdi_2_5 = statistics['hdi_2.5%'].mu
            mu_hdi_97_5 = statistics['hdi_97.5%'].mu
            scale_hdi_2_5 = statistics['hdi_2.5%'][self.scale_name]
            scale_hdi_97_5 = statistics['hdi_97.5%'][self.scale_name]

            if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5,
                                       real_scale, scale_hdi_2_5, scale_hdi_97_5):
                correct += 1

        assert correct >= 95, f"Coverage for fixed mu=0 was {correct}%, expected >=95%"

    def test_degradation(self):
        """
        Check that coverage improves (or does not degrade) as the number of
        MCMC samples increases.
        """
        n_simulations = [x0, x1, x2]
        real_mu = 0
        real_scale = self.real_scale_fixed

        mu_prior = Uniform(0, 2 * np.pi)
        model = self.model_factory(mu_prior, self.scale_prior)
        service = BayesianInferenceService(model)

        # Generate the same set of datasets for all simulation sizes
        samples_list = []
        for _ in range(self.n_cycles):
            samples_list.append(self.dist_rvs(**{self.scale_name: real_scale,
                                                 'mu': real_mu,
                                                 'size': self.sample_size}))

        coverage_10 = coverage_100 = coverage_10000 = 0.0

        for n_sim in n_simulations:
            correct = 0
            for sample in samples_list:
                data = {'N': self.sample_size, 'values': sample}

                posterior = service.build_model(data)
                fit = service.get_samples(posterior, sample_amount=n_sim)
                statistics = service.get_statistics(fit, confidence_interval=95)

                mu_hdi_2_5 = statistics['hdi_2.5%'].mu
                mu_hdi_97_5 = statistics['hdi_97.5%'].mu
                scale_hdi_2_5 = statistics['hdi_2.5%'][self.scale_name]
                scale_hdi_97_5 = statistics['hdi_97.5%'][self.scale_name]

                if correct_inferred_values(real_mu, mu_hdi_2_5, mu_hdi_97_5,
                                           real_scale, scale_hdi_2_5, scale_hdi_97_5):
                    correct += 1

            coverage = correct / self.n_cycles
            if n_sim == x0:
                coverage_10 = coverage
            elif n_sim == x1:
                coverage_100 = coverage
            elif n_sim == x2:
                coverage_10000 = coverage

        # Check monotonic improvement (or at least non‑degradation)
        assert coverage_10 <= coverage_100 <= coverage_10000, \
            f"Coverage not monotonic: {coverage_10:.3f} -> {coverage_100:.3f} -> {coverage_10000:.3f}"


# ----------------------------------------------------------------------
# Concrete test classes for each distribution
# ----------------------------------------------------------------------

class TestVonMises(BaseCircularTest):
    """Tests for the von Mises distribution."""
    dist_rvs = staticmethod(vonmises.rvs)
    scale_prior = Exponential(0.1)
    scale_name = 'kappa'
    model_factory = VonMisesMK
    real_scale_fixed = 2
    real_scale_random_range = (0, 7)

class TestCardioid(BaseCircularTest):
    """Tests for the Cardioid distribution."""
    dist_rvs = staticmethod(cardioid.rvs)
    scale_prior = Normal(0.25, 0.25)
    scale_name = 'rho'
    model_factory = Cardioid
    real_scale_fixed = 0.3
    real_scale_random_range = (0, 0.5)

class TestWrappedCauchy(BaseCircularTest):
    """Tests for the wrapped Cauchy distribution."""
    dist_rvs = staticmethod(wrapcauchy.rvs)
    scale_prior = Normal(0.5, 0.5)
    scale_name = 'rho'
    model_factory = WrappedCauchy
    real_scale_fixed = 0.7
    real_scale_random_range = (0, 1)

# enquanto faço a documentação fazer um draft(esboço do artigo (documentação da lib como artigo)) para o journal of statistical software
