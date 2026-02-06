import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

class TestGetPercentiles:
    def test_default_values(self, service):
        # create 101 elementes from 0 to 100
        data = np.arange(101)

        result = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=2.5,
            round_to=2
        )

        assert isinstance(result, pd.Series)

        assert "hdi_2.5%" in result.index
        assert "hdi_97.5%" in result.index

        assert pytest.approx(result["hdi_2.5%"], 0.01) == 2.5
        assert pytest.approx(result["hdi_97.5%"], 0.01) == 97.5

    def test_percentiles_with_50_interval(self, service):
        # test percentile calculation for 50% interval (median only)
        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=50,
            round_to=1
        )
        
        assert "hdi_50%" in result.index

        assert result["hdi_50%"] == 5
    
    @pytest.mark.parametrize("interval,expected_low,expected_high", [
        (1, 1.0, 99.0),     # 1st and 99th percentiles
        (5, 5.0, 95.0),     # 5th and 95th percentiles
        (10, 10.0, 90.0),   # 10th and 90th percentiles
        (25, 25.0, 75.0),   # 25th and 75th percentiles
        (40, 40.0, 60.0),   # 40th and 60th percentiles
    ])
    def test_different_confidence_intervals(self, service, interval, expected_low, expected_high):
        """Test various symmetric confidence intervals"""
        data = np.arange(101)  # 0 to 100
        
        result = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=interval,
            round_to=1
        )
        
        low_key = f"hdi_{interval}%"
        high_key = f"hdi_{100-interval}%"
        
        assert low_key in result.index
        assert high_key in result.index
        assert pytest.approx(result[low_key], 0.1) == expected_low
        assert pytest.approx(result[high_key], 0.1) == expected_high
    
    def test_asymmetric_input_handling(self, service):
        """Test that lower bound is always on left, even if input > 50"""
        data = np.arange(101)
        
        # Input 97.5 should be treated as 2.5
        result_high = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=97.5,
            round_to=2
        )
        
        # Input 2.5 should give same result
        result_low = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=2.5,
            round_to=2
        )
        
        # Both should have same keys and values
        assert set(result_high.index) == set(result_low.index)
        assert pytest.approx(result_high["hdi_2.5%"], 0.01) == result_low["hdi_2.5%"]
        assert pytest.approx(result_high["hdi_97.5%"], 0.01) == result_low["hdi_97.5%"]

    def test_rounding_behavior(self, service):
        """Test different rounding precisions"""
        data = np.array([1.123456, 2.234567, 3.345678])
        
        # Test with 0 decimal places
        result_0 = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=10,
            round_to=0
        )
        assert result_0["hdi_10%"] == 1
        assert result_0["hdi_90%"] == 3
        
        # Test with 2 decimal places
        result_2 = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=10,
            round_to=2
        )
        hdi_10 = data[0] + ((data[-1] - data[0]) * 0.1)
        hdi_90 = data[0] + ((data[-1] - data[0]) * 0.9)
        assert result_2["hdi_10%"] == np.round(hdi_10, 2)
        assert result_2["hdi_90%"] == np.round(hdi_90, 2)
        
        # Test with 4 decimal places
        result_4 = service._BayesianInferenceService__get_percentiles(
            param=data,
            confidence_interval=10,
            round_to=4
        )
        hdi_10 = data[0] + ((data[-1] - data[0]) * 0.1)
        hdi_90 = data[0] + ((data[-1] - data[0]) * 0.9)
        assert result_4["hdi_10%"] == np.round(hdi_10, 4)
        assert result_4["hdi_90%"] == np.round(hdi_90, 4)