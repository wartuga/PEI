# tests/service/test_get_values.py
import pytest
import json
from unittest.mock import Mock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from stan_circular_inference.service.bayesian_inference import BayesianInferenceService


class TestGetValues:
    """Test suite para o método get_values"""
    
    def test_get_values_single_parameter(self, service, mock_fit_with_chains):
        """Testa extração de um único parâmetro"""
        # Arrange
        parameters = ["mu"]
        
        # Act
        result = service.get_values(mock_fit_with_chains, parameters)
        
        # Assert
        assert isinstance(result, dict)
        assert "mu" in result
        assert len(result["mu"]) == 5  # 3 da chain 1 + 2 da chain 2
        
        # Verifica valores específicos
        expected_mu_values = [3.14, 3.15, 3.16, 3.17, 3.18]
        for expected, actual in zip(expected_mu_values, result["mu"]):
            assert pytest.approx(actual, 0.001) == expected
    
    def test_get_values_multiple_parameters(self, service, mock_fit_with_chains):
        """Testa extração de múltiplos parâmetros"""
        # Arrange
        parameters = ["mu", "kappa"]
        
        # Act
        result = service.get_values(mock_fit_with_chains, parameters)
        
        # Assert
        assert set(result.keys()) == {"mu", "kappa"}
        
        # Verifica todos têm o mesmo número de valores
        assert len(result["mu"]) == len(result["kappa"]) == 5
        
        # Verifica alguns valores
        assert pytest.approx(result["mu"][0], 0.001) == 3.14
        assert pytest.approx(result["kappa"][0], 0.001) == 1.0
        
        assert pytest.approx(result["mu"][-1], 0.001) == 3.18
        assert pytest.approx(result["kappa"][-1], 0.001) == 1.4
    
    def test_get_values_ignores_non_sample_topics(self, service, mock_fit_with_chains):
        """Testa que apenas lines com topic='sample' são processadas"""
        # Na chain 1 temos 4 lines, mas 1 tem topic='info', então apenas 3 samples
        # Na chain 2 temos 2 samples
        # Total esperado: 3 + 2 = 5
        
        parameters = ["mu"]
        result = service.get_values(mock_fit_with_chains, parameters)
        
        assert len(result["mu"]) == 5  # Ignora o 'info'
    
    def test_get_values_single_chain(self, service, mock_fit_single_chain):
        """Testa com apenas uma chain"""
        # Arrange
        parameters = ["mu"]
        
        # Act
        result = service.get_values(mock_fit_single_chain, parameters)
        
        # Assert
        assert result["mu"] == [1.0, 2.0, 3.0]
    
    def test_get_values_parameter_not_in_all_samples(self, service):
        """Testa quando um parâmetro não está presente em todos os samples"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": {"mu": 1.0, "kappa": 0.5}},
            {"topic": "sample", "values": {"mu": 2.0}},  # Sem kappa
            {"topic": "sample", "values": {"mu": 3.0, "kappa": 0.7}},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu", "kappa"]
        
        # Act & Assert - deve lançar KeyError quando kappa não está presente
        with pytest.raises(KeyError):
            service.get_values(fit, parameters)
    
    def test_get_values_with_nested_values(self, service):
        """Testa com values que são dicionários (isinstance(values, dict))"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": {"mu": {"value": 1.0, "sd": 0.1}}},
            {"topic": "sample", "values": {"mu": {"value": 2.0, "sd": 0.2}}},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert
        assert result["mu"] == [{"value": 1.0, "sd": 0.1}, {"value": 2.0, "sd": 0.2}]
    
    def test_get_values_empty_parameter_list(self, service, mock_fit_with_chains):
        """Testa com lista vazia de parâmetros"""
        # Arrange
        parameters = []
        
        # Act
        result = service.get_values(mock_fit_with_chains, parameters)
        
        # Assert
        assert result == {}
    
    def test_get_values_single_parameter_not_present(self, service, mock_fit_with_chains):
        """Testa quando o parâmetro solicitado não existe nas chains"""
        # Arrange
        parameters = ["non_existent_param"]
        
        # Act & Assert - deve lançar KeyError
        with pytest.raises(KeyError):
            service.get_values(mock_fit_with_chains, parameters)
    
    def test_get_values_decoding_error(self, service):
        """Testa com chains que não podem ser decodificadas"""
        # Cria chains com bytes inválidos
        chains = [
            b'\xff\xfe'  # Bytes inválidos para UTF-8
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act & Assert - deve lançar UnicodeDecodeError
        with pytest.raises(UnicodeDecodeError):
            service.get_values(fit, parameters)
    
    def test_get_values_json_parse_error(self, service):
        """Testa com JSON inválido nas chains"""
        # Cria chains com JSON inválido
        chains = [b'{"invalid": json}']  # JSON inválido
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act & Assert - deve lançar json.JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            service.get_values(fit, parameters)
    
    def test_get_values_values_not_dict(self, service):
        """Testa quando values não é um dicionário"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": "not a dict"},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert - deve pular este sample (isinstance(values, dict) é False)
        assert result["mu"] == []
    
    def test_get_values_whitespace_handling(self, service):
        """Testa que whitespace é tratado corretamente (strip())"""
        # Cria chain com whitespace - cria como string primeiro
        json_str = json.dumps({"topic": "sample", "values": {"mu": 1.0}})
        chain_str = '  \n  ' + json_str + '  \n  '
        
        chains = [
            chain_str.encode('utf-8')
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert - strip() deve remover whitespace
        assert result["mu"] == [1.0]
    
    def test_get_values_order_preserved(self, service):
        """Testa que a ordem dos valores é preservada"""
        # Dados das chains
        chain1_data = [
            {"topic": "sample", "values": {"mu": 1.0}},
            {"topic": "sample", "values": {"mu": 2.0}},
            {"topic": "sample", "values": {"mu": 3.0}},
        ]
        
        chain2_data = [
            {"topic": "sample", "values": {"mu": 4.0}},
            {"topic": "sample", "values": {"mu": 5.0}},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain1_data]),
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain2_data]),
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert - ordem deve ser preservada (chain 1 primeiro, depois chain 2)
        assert result["mu"] == [1.0, 2.0, 3.0, 4.0, 5.0]


class TestGetValuesEdgeCases:
    """Testes de casos extremos para get_values"""
    
    @pytest.fixture
    def service(self):
        mock_model = Mock()
        mock_model.gen_stan_model.return_value = "test_code"
        return BayesianInferenceService(mock_model)
    
    def test_get_values_large_chains(self, service):
        """Testa com chains muito grandes"""
        # Cria 1000 samples por chain
        samples_per_chain = 1000
        chains = []
        
        for chain_idx in range(4):  # 4 chains
            chain_samples = []
            for sample_idx in range(samples_per_chain):
                mu_value = chain_idx * 1000 + sample_idx
                chain_samples.append(
                    json.dumps({"topic": "sample", "values": {"mu": float(mu_value)}})
                )
            # Converte cada string para bytes individualmente antes do join
            chains.append(
                b'\n'.join([item.encode('utf-8') for item in chain_samples])
            )
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert
        assert len(result["mu"]) == 4 * samples_per_chain  # 4000 samples total
        
        # Verifica alguns valores
        assert result["mu"][0] == 0.0  # chain 0, sample 0
        assert result["mu"][1000] == 1000.0  # chain 1, sample 0
        assert result["mu"][-1] == 3999.0  # chain 3, último sample
    
    def test_get_values_null_values(self, service):
        """Testa com valores null/none no JSON"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": {"mu": None}},
            {"topic": "sample", "values": {"mu": 1.0}},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert
        assert result["mu"] == [None, 1.0]
    
    def test_get_values_special_characters(self, service):
        """Testa com caracteres especiais nos parâmetros"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": {"mu.beta": 1.0, "sigma_alpha": 2.0}},
            {"topic": "sample", "values": {"mu.beta": 3.0, "sigma_alpha": 4.0}},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu.beta", "sigma_alpha"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert
        assert "mu.beta" in result
        assert "sigma_alpha" in result
        assert result["mu.beta"] == [1.0, 3.0]
        assert result["sigma_alpha"] == [2.0, 4.0]
    
    def test_get_values_mixed_data_types(self, service):
        """Testa com diferentes tipos de dados nos valores"""
        # Dados da chain
        chain_data = [
            {"topic": "sample", "values": {
                "mu": 1.0,  # float
                "count": 42,  # int
                "name": "test",  # string
                "flag": True,  # boolean
                "array": [1, 2, 3]  # array
            }},
        ]
        
        # Converte para bytes
        chains = [
            b'\n'.join([json.dumps(item).encode('utf-8') for item in chain_data])
        ]
        
        fit = Mock()
        fit.stan_outputs = chains
        
        parameters = ["mu", "count", "name", "flag", "array"]
        
        # Act
        result = service.get_values(fit, parameters)
        
        # Assert
        assert result["mu"] == [1.0]
        assert result["count"] == [42]
        assert result["name"] == ["test"]
        assert result["flag"] == [True]
        assert result["array"] == [[1, 2, 3]]