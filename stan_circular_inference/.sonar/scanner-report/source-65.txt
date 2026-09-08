import warnings
import pytest
import json
import pandas as pd
import numpy as np
import arviz as az

# Suppress all warnings from stan library
warnings.filterwarnings("ignore")

from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stan_circular_inference.service.bayesian_inference import BayesianInferenceService

example_model = '''
    data {
        int<lower=0> N;
        vector[N] values;
    }

    parameters {
        real<lower=0, upper=2*pi()> mu;
        real<lower=0> kappa;
    }

    model {
        mu ~ uniform(0, 2*pi());
        kappa ~ exponential(0.1);
        
        values ~ von_mises(mu, kappa);
    }
'''

@pytest.fixture
def bayesian_service():
    return BayesianInferenceService(example_model)

@pytest.fixture
def mock_stan_build():
    """Fixture para mockar stan.build"""
    with patch('stan_circular_inference.service.bayesian_inference.stan.build') as mock:
        yield mock

@pytest.fixture
def mock_posterior():
    """Fixture para criar um mock do posterior (usado em get_samples)"""
    posterior = Mock()
    fit = Mock()
    posterior.sample = Mock(return_value=fit)
    return posterior

@pytest.fixture
def mock_fit_with_chains():
    """Cria um mock de fit com chains no formato esperado"""
    # Dados das chains para get_values
    chain1_data = [
        {"topic": "sample", "values": {"mu": 3.14, "kappa": 1.0}},
        {"topic": "sample", "values": {"mu": 3.15, "kappa": 1.1}},
        {"topic": "info", "values": "some info"},
        {"topic": "sample", "values": {"mu": 3.16, "kappa": 1.2}},
    ]
    
    chain2_data = [
        {"topic": "sample", "values": {"mu": 3.17, "kappa": 1.3}},
        {"topic": "sample", "values": {"mu": 3.18, "kappa": 1.4}},
    ]
    
    # Converte para bytes
    chains = [
        b'\n'.join([json.dumps(item).encode('utf-8') for item in chain1_data]),
        b'\n'.join([json.dumps(item).encode('utf-8') for item in chain2_data]),
    ]

    return MockFit(chains, mu_values=[3.14, 3.15, 3.16, 3.17, 3.18], kappa_values=[1.0, 1.1, 1.2, 1.3, 1.4])

@pytest.fixture
def mock_fit_single_chain():
    """Cria um mock de fit com apenas uma chain"""
    # Dados da chain para get_values
    chain_data = [
        {"topic": "sample", "values": {"mu": 1.0, "kappa": 1.1}},
        {"topic": "sample", "values": {"mu": 2.0, "kappa": 1.2}},
        {"topic": "sample", "values": {"mu": 3.0, "kappa": 1.3}},
    ]
    
    chains = [b'\n'.join(json.dumps(item).encode('utf-8') for item in chain_data)]
    
    return MockFit(chains, mu_values=[1.0, 2.0, 3.0], kappa_values=[1.1, 1.2, 1.3])

@pytest.fixture
def mock_az_data():
    """Create a real (minimal) InferenceData object for testing."""
    # Simulate posterior draws for two parameters
    posterior = {
        "mu": np.random.normal(3.14, 0.1, size=(4, 100)),   # (chains, draws)
        "kappa": np.random.gamma(2, 0.5, size=(4, 100))
    }
    coords = {"chain": [0,1,2,3], "draw": np.arange(100)}
    return az.from_dict(posterior=posterior, coords=coords)

class MockFit:
    def __init__(self, chains, mu_values, kappa_values):
        self.stan_outputs = chains
        self._param_data = {
            "mu": np.array(mu_values),
            "kappa": np.array(kappa_values)
        }
    
    def keys(self):
        return list(self._param_data.keys())
    
    def __getitem__(self, key):
        if key in self._param_data:
            return self._param_data[key]
        raise KeyError(f"Parameter '{key}' not found")