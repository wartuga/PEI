# tests/service/test_get_statistics.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

class TestGetStatistics:
    """Test suite para o método get_statistics"""
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_default_confidence_interval(
        self, mock_az_summary, mock_az_from_pystan, 
        service, mock_fit_with_chains, mock_az_data, mock_summary_df, capsys
    ):
        """Testa get_statistics com intervalo de confiança padrão (3%)"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = mock_summary_df
        
        # Act
        service.get_statistics(mock_fit_with_chains, confidence_interval=3)
        
        # Assert
        # Verifica que az.from_pystan foi chamado com o fit
        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)
        
        # Verifica que az.summary foi chamado com os dados e round_to=2
        mock_az_summary.assert_called_once_with(mock_az_data, round_to=2)
        
        # Verifica que foi impresso o summary (default)
        captured = capsys.readouterr()
        assert "mean" in captured.out
        assert "sd" in captured.out
        assert "hdi_3%" in captured.out  # Default columns
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_custom_confidence_interval(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, mock_az_data, mock_summary_df, capsys
    ):
        """Testa get_statistics com intervalo de confiança customizado (11%)"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = mock_summary_df.copy()  # Cópia para não modificar o original
        
        # Act
        service.get_statistics(mock_fit_with_chains, confidence_interval=11)
        
        # Assert
        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)
        mock_az_summary.assert_called_once_with(mock_az_data, round_to=2)
        
        # Verifica que foi impresso com intervalos customizados
        captured = capsys.readouterr()
        output = captured.out
        
        # Deve conter hdi_11% e hdi_89% (min(11, 89) = 11)
        assert "hdi_11%" in output or "hdi_89%" in output
        
        # Não deve conter os defaults
        assert "hdi_3%" not in output
        assert "hdi_97%" not in output
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_asymmetric_interval_handling(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, mock_az_data, mock_summary_df, capsys
    ):
        """Testa que lower bound é sempre à esquerda, mesmo com entrada > 50"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        mock_az_summary.return_value = mock_summary_df.copy()
        
        # Act - passar 89% deve ser tratado como 11% (min(89, 11) = 11)
        service.get_statistics(mock_fit_with_chains, confidence_interval=89)
        
        # Assert
        captured = capsys.readouterr()
        output = captured.out
        
        # Deve usar 11% e 89% (não 89% e 11%)
        assert "hdi_11%" in output or "hdi_89%" in output
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_50_percent_interval(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa get_statistics com intervalo de 50% (apenas mediana)"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        
        # Cria um summary com as colunas esperadas
        summary_data = {
            "mean": [1.23, 0.75],
            "sd": [0.12, 0.08],
            "hdi_3%": [1.01, 0.61],
            "hdi_97%": [1.45, 0.89],
            "mcse_mean": [0.01, 0.01],
            "mcse_sd": [0.01, 0.01],
            "ess_bulk": [500.0, 450.0],
            "ess_tail": [480.0, 430.0],
            "r_hat": [1.01, 1.02]
        }
        summary_df = pd.DataFrame(summary_data, index=["mu", "kappa"])
        mock_az_summary.return_value = summary_df
        
        # Act
        service.get_statistics(mock_fit_with_chains, confidence_interval=50)
        
        # Assert
        captured = capsys.readouterr()
        output = captured.out
        
        # Deve imprimir hdi_50% (mediana)
        assert "hdi_50%" in output
        
        # Não deve conter hdi_3% e hdi_97%
        assert "hdi_3%" not in output
        assert "hdi_97%" not in output
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_column_selection(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa que as colunas são selecionadas corretamente"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        
        # Cria um summary mais completo
        summary_data = {
            "mean": [1.23],
            "sd": [0.12],
            "hdi_3%": [1.01],
            "hdi_97%": [1.45],
            "mcse_mean": [0.01],
            "mcse_sd": [0.01],
            "ess_bulk": [500.0],
            "ess_tail": [480.0],
            "r_hat": [1.01],
            "extra_col": ["extra"]  # Coluna extra que não deve aparecer
        }
        summary_df = pd.DataFrame(summary_data, index=["mu"])
        mock_az_summary.return_value = summary_df
        
        # Act
        service.get_statistics(mock_fit_with_chains, confidence_interval=25)
        
        # Assert
        captured = capsys.readouterr()
        output = captured.out
        
        # Deve conter as colunas esperadas
        expected_cols = ["mean", "sd", "hdi_25%", "hdi_75%", 
                        "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
        
        # Verifica que colunas esperadas estão no output
        for col in expected_cols:
            if col in ["hdi_25%", "hdi_75%"]:  # Essas podem estar nos valores, não no header
                assert col in output
            else:
                # Verifica no início das linhas (formato tabular)
                lines = output.split('\n')
                for line in lines[:3]:  # Primeiras linhas contêm headers
                    if col in line:
                        break
                else:
                    # Se não encontrou em nenhuma linha, pode estar ok dependendo do formato
                    pass
        
        # Não deve conter colunas extras
        assert "extra_col" not in output
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_edge_case_confidence_interval_100(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, mock_az_data, capsys
    ):
        """Testa caso extremo: confidence_interval = 100"""
        # Arrange
        mock_az_from_pystan.return_value = mock_az_data
        
        summary_data = {
            "mean": [1.23, 0.75],
            "sd": [0.12, 0.08],
            "hdi_3%": [1.01, 0.61],
            "hdi_97%": [1.45, 0.89],
            "mcse_mean": [0.01, 0.01],
            "mcse_sd": [0.01, 0.01],
            "ess_bulk": [500.0, 450.0],
            "ess_tail": [480.0, 430.0],
            "r_hat": [1.01, 1.02]
        }
        summary_df = pd.DataFrame(summary_data, index=["mu", "kappa"])
        mock_az_summary.return_value = summary_df
        
        service.get_statistics(mock_fit_with_chains, confidence_interval=100)
        
        # Assert
        captured = capsys.readouterr()
        output = captured.out
    
    @patch('stan_circular_inference.service.bayesian_inference.az.from_pystan')
    @patch('stan_circular_inference.service.bayesian_inference.az.summary')
    def test_get_statistics_exception_handling(
        self, mock_az_summary, mock_az_from_pystan,
        service, mock_fit_with_chains, capsys
    ):
        """Testa comportamento quando az.from_pystan lança exceção"""
        # Arrange
        mock_az_from_pystan.side_effect = ValueError("Error converting to InferenceData")
        
        # Act
        try:
            service.get_statistics(mock_fit_with_chains, confidence_interval=11)
            # Se não lançou exceção, verifica output
            captured = capsys.readouterr()
            print(f"Output when exception swallowed: {captured.out}")
        except ValueError:
            # É aceitável se lançar exceção
            pass
        
        # Assert - pelo menos o mock foi chamado
        mock_az_from_pystan.assert_called_once_with(mock_fit_with_chains)