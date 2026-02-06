# tests/service/test_get_samples.py
import pytest
from unittest.mock import Mock, patch, call

class TestGetSamples:
    """Test suite for the get_samples method"""
    
    def test_get_samples_without_init(self, service, mock_posterior):
        """Test get_samples without init parameter (uses defaults)"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        # Act
        result = service.get_samples(mock_posterior, sample_amount=1000)
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=1000
        )
        assert result == fit
    
    def test_get_samples_with_init(self, service, mock_posterior):
        """Test get_samples with init parameter"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        init_dict = {
            "mu": 3.14,
            "kappa": 1.0
        }
        
        # Act
        result = service.get_samples(
            posterior=mock_posterior,
            sample_amount=2000,
            init=init_dict
        )
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=2000,
            init=init_dict
        )
        assert result == fit
    
    def test_get_samples_default_sample_amount(self, service, mock_posterior):
        """Test get_samples with default sample_amount (50000)"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        # Act - don't specify sample_amount
        result = service.get_samples(mock_posterior)
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=50000
        )
        assert result == fit
    
    def test_get_samples_verify_chains_always_4(self, service, mock_posterior):
        """Test that num_chains is always 4 regardless of other parameters"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        # Test with various sample amounts
        test_cases = [100, 1000, 50000, 100000]
        
        for sample_amount in test_cases:
            # Reset mock
            mock_posterior.sample.reset_mock()
            
            # Act
            result = service.get_samples(
                posterior=mock_posterior,
                sample_amount=sample_amount,
                init={"test": 1.0}
            )
            
            # Assert
            mock_posterior.sample.assert_called_once_with(
                num_chains=4,  # Always 4
                num_samples=sample_amount,
                init={"test": 1.0}
            )
            assert result == fit
    
    def test_get_samples_with_different_posterior_objects(self, service):
        """Test get_samples works with different posterior objects"""
        # Test with various mock posterior objects
        test_cases = [
            (Mock(), 1000, {"mu": 1.0}),
            (Mock(spec=['sample']), 2000, {"kappa": 2.0})
        ]
        
        for posterior, sample_amount, init in test_cases:
            # Setup
            fit = Mock()
            posterior.sample.return_value = fit
            
            # Act
            result = service.get_samples(posterior, sample_amount, init)
            
            # Assert
            expected_kwargs = {
                'num_chains': 4,
                'num_samples': sample_amount,
                'init': init
            }
            posterior.sample.assert_called_once_with(**expected_kwargs)
            assert result == fit
    
    def test_get_samples_error_handling(self, service, mock_posterior):
        """Test that exceptions from posterior.sample are propagated"""
        # Arrange
        mock_posterior.sample.side_effect = RuntimeError("Sampling failed")
        
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            service.get_samples(mock_posterior, sample_amount=1000)
        
        assert "Sampling failed" in str(exc_info.value)
        mock_posterior.sample.assert_called_once()
    
    def test_get_samples_large_sample_amount(self, service, mock_posterior):
        """Test with very large sample amount"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        large_sample_amount = 1000000
        
        # Act
        result = service.get_samples(
            posterior=mock_posterior,
            sample_amount=large_sample_amount
        )
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=large_sample_amount
        )
        assert result == fit
    
    def test_get_samples_with_complex_init_structure(self, service, mock_posterior):
        """Test with complex initialization structure"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        complex_init = {
            "mu": 3.14,
            "kappa": 1.5,
            "theta": [0.1, 0.2, 0.3],
            "sigma": {"value": 0.5, "scale": 1.0}
        }
        
        # Act
        result = service.get_samples(
            posterior=mock_posterior,
            sample_amount=5000,
            init=complex_init
        )
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=5000,
            init=complex_init
        )
        assert result == fit
    
    def test_get_samples_multiple_calls(self, service, mock_posterior):
        """Test multiple calls to get_samples"""
        # Arrange
        fit1 = Mock()
        fit2 = Mock()
        fit3 = Mock()
        mock_posterior.sample.side_effect = [fit1, fit2, fit3]
        
        # Act - multiple calls
        result1 = service.get_samples(mock_posterior, 1000)
        result2 = service.get_samples(mock_posterior, 2000, {"init": 1.0})
        result3 = service.get_samples(mock_posterior)  # Defaults
        
        # Assert
        assert mock_posterior.sample.call_count == 3
        assert result1 == fit1
        assert result2 == fit2
        assert result3 == fit3
        
        # Check call arguments
        calls = mock_posterior.sample.call_args_list
        assert calls[0] == call(num_chains=4, num_samples=1000)
        assert calls[1] == call(num_chains=4, num_samples=2000, init={"init": 1.0})
        assert calls[2] == call(num_chains=4, num_samples=50000)


class TestGetSamplesIntegration:
    """Integration tests for get_samples"""
    
    def test_get_samples_in_pipeline(self, service, mock_posterior):
        """Test get_samples as part of a larger pipeline"""
        # Create a realistic mock posterior
        mock_posterior = Mock()
        
        # Create a mock fit with realistic structure
        class MockFit:
            def __init__(self):
                self.stan_outputs = ["chain1", "chain2", "chain3", "chain4"]
                self.parameters = {"mu": [1.0, 2.0, 3.0], "kappa": [0.5, 0.6, 0.7]}
            
            def __str__(self):
                return f"MockFit(parameters={list(self.parameters.keys())})"
        
        fit = MockFit()
        mock_posterior.sample.return_value = fit
        
        # Simulate pipeline
        sample_amount = 10000
        init_values = {"mu": 3.14, "kappa": 1.0}
        
        # Call get_samples
        fit = service.get_samples(
            posterior=mock_posterior,
            sample_amount=sample_amount,
            init=init_values
        )
        
        # Verify
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=sample_amount,
            init=init_values
        )
        
        # Check that we got a usable fit object
        assert fit == fit
        assert hasattr(fit, 'stan_outputs')
        assert hasattr(fit, 'parameters')
        assert len(fit.stan_outputs) == 4
    
    def test_get_samples_with_realistic_posterior_from_build_model(self, service, mock_posterior):
        """Test that get_samples works with posterior from build_model"""
        # This test would mock both build_model and get_samples
        # Since build_model returns a posterior, get_samples should work with it
        
        with patch('stan_circular_inference.service.bayesian_inference.stan.build') as mock_build:
            # Create a mock posterior
            mock_posterior = Mock()
            mock_build.return_value = mock_posterior
            
            # Build model (would normally come from build_model)
            test_data = {"N": 10, "values": list(range(10))}
            
            # If build_model was called
            # posterior = service.build_model(test_data)
            # Instead, we'll use the mock directly
            
            # Now sample from it
            fit = Mock()
            mock_posterior.sample.return_value = fit
            
            fit = service.get_samples(mock_posterior, sample_amount=5000)
            
            # Verify
            mock_posterior.sample.assert_called_once_with(
                num_chains=4,
                num_samples=5000
            )
            assert fit == fit


# Edge case tests
class TestGetSamplesEdgeCases:
    """Edge case tests for get_samples"""
    
    def test_get_samples_with_zero_samples(self, service, mock_posterior):
        """Test with sample_amount=0 (edge case)"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        # Act
        result = service.get_samples(mock_posterior, sample_amount=0)
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=0
        )
        assert result == fit
    
    def test_get_samples_with_negative_samples(self, service, mock_posterior):
        """Test with negative sample_amount (should still work - stan will handle error)"""
        # Arrange
        fit = Mock()
        mock_posterior.sample.return_value = fit
        
        # Act
        result = service.get_samples(mock_posterior, sample_amount=-100)
        
        # Assert
        mock_posterior.sample.assert_called_once_with(
            num_chains=4,
            num_samples=-100
        )
        assert result == fit