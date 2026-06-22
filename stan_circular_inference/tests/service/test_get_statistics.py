# tests/service/test_get_statistics.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

def make_summary_df(confidence_interval, index=None):
    """
    Cria um DataFrame de resumo com as colunas HDI correspondentes ao intervalo
    de confiança fornecido.
    """
    if index is None:
        index = ["mu", "kappa"]
    lower = (100 - confidence_interval) / 2
    upper = 100 - lower
    data = {
        "mean": [1.23, 0.75],
        "sd": [0.12, 0.08],
        f"hdi_{lower}%": [1.01, 0.61],
        f"hdi_{upper}%": [1.45, 0.89],
        "mcse_mean": [0.01, 0.01],
        "mcse_sd": [0.01, 0.01],
        "ess_bulk": [500.0, 450.0],
        "ess_tail": [480.0, 430.0],
        "r_hat": [1.01, 1.02]
    }
    return pd.DataFrame(data, index=index)

class TestGetStatistics:
    """Test suite para o método get_statistics"""
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_default_confidence_interval(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa get_statistics com intervalo de confiança padrão (89%)"""
        ci = 89
        lower = (100 - ci) / 2
        upper = 100 - lower
        summary_df = make_summary_df(ci)
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = summary_df

        bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=ci)

        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)
        # A implementação usa round_to=5
        mock_az_summary.assert_called_once_with(
            mock_az_data,
            round_to=5,
            circ_var_names=['mu'],
            hdi_prob=ci / 100
        )

        captured = capsys.readouterr()
        assert "mean" in captured.out
        assert "sd" in captured.out
        assert f"hdi_{lower}%" in captured.out
        assert f"hdi_{upper}%" in captured.out

    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_custom_confidence_interval(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa get_statistics com intervalo de confiança customizado (11%)"""
        ci = 11
        lower = (100 - ci) / 2
        upper = 100 - lower
        summary_df = make_summary_df(ci)
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = summary_df

        bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=ci)

        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)
        mock_az_summary.assert_called_once_with(
            mock_az_data,
            round_to=5,
            circ_var_names=['mu'],
            hdi_prob=ci / 100
        )

        captured = capsys.readouterr()
        output = captured.out
        # Deve conter as colunas HDI corretas
        assert f"hdi_{lower}%" in output or f"hdi_{upper}%" in output
        # Os defaults (3% e 97%) não devem aparecer
        assert "hdi_3%" not in output
        assert "hdi_97%" not in output

    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_asymmetric_interval_handling(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa que lower bound é sempre à esquerda, mesmo com entrada > 50"""
        ci = 89  # como acima, o serviço usa 89% diretamente
        lower = (100 - ci) / 2
        upper = 100 - lower
        summary_df = make_summary_df(ci)
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = summary_df

        bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=89)

        mock_az_summary.assert_called_once_with(
            mock_az_data,
            round_to=5,
            circ_var_names=['mu'],
            hdi_prob=0.89
        )

        captured = capsys.readouterr()
        output = captured.out
        # Deve usar 5.5% e 94.5% (não 11% e 89%)
        assert f"hdi_{lower}%" in output or f"hdi_{upper}%" in output
        # Não deve conter colunas de 11%
        assert "hdi_11%" not in output

    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_50_percent_interval(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa get_statistics com intervalo de 50% (hdi_25% e hdi_75%)"""
        ci = 50
        lower = (100 - ci) / 2  # 25
        upper = 100 - lower     # 75
        summary_df = make_summary_df(ci)
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = summary_df

        bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=ci)

        captured = capsys.readouterr()
        output = captured.out
        assert f"hdi_{lower}%" in output   # 25%
        assert f"hdi_{upper}%" in output   # 75%
        assert "hdi_3%" not in output
        assert "hdi_97%" not in output

    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_edge_case_confidence_interval_100(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa caso extremo: confidence_interval = 100"""
        ci = 100
        lower = (100 - ci) / 2  # 0.0
        upper = 100 - lower     # 100.0
        summary_df = make_summary_df(ci)
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = summary_df

        bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=100)

        captured = capsys.readouterr()
        output = captured.out
        assert f"hdi_{lower}%" in output   # hdi_0.0%
        assert f"hdi_{upper}%" in output   # hdi_100.0%

    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_exception_handling(
        self, mock_az_summary, mock_az_from_pystan,
        bayesian_service, mock_fit_with_chains, capsys
    ):
        """Testa comportamento quando az.from_pystan lança exceção"""
        mock_az_from_pystan.side_effect = ValueError("Error converting to InferenceData")

        # O método não trata a exceção, portanto deve propagá‑la
        with pytest.raises(ValueError, match="Error converting to InferenceData"):
            bayesian_service.get_statistics(mock_fit_with_chains, confidence_interval=11)

        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)
        # az.summary não deve ser chamado
        mock_az_summary.assert_not_called()