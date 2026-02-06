import pytest
from unittest.mock import Mock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stan_circular_inference.service.bayesian_inference import BayesianInferenceService

class TestBuildModel:
    """Test suite para o método build_model"""
    
    def test_build_model_success_with_print(self, mock_stan_build, service, capsys):
        """Testa build_model bem-sucedido com prints"""
        # Arrange
        model = Mock()
        mock_stan_build.return_value = model
        
        test_data = {
            "N": 5,
            "values": [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        
        # Act
        result = service.build_model(test_data)
        
        # Assert
        mock_stan_build.assert_called_once_with(
            service.model,  # This is the example_model from your fixture
            data=test_data
        )
        assert result == model
    
    def test_build_model_type_error(self, mock_stan_build, service, capsys):
        """Testa build_model com TypeError"""
        # Arrange
        mock_stan_build.side_effect = TypeError("Original error message")
        
        invalid_data = "not_a_dict"  # Tipo errado
        
        # Act & Assert
        with pytest.raises(TypeError) as exc_info:
            service.build_model(invalid_data)
        
        # Verifica mensagem de erro personalizada
        assert str(exc_info.value) == "Wrong parameter format for the model!"
        
        # Verifica que NÃO houve prints (já que entrou no except)
        captured = capsys.readouterr()
        assert captured.out == ""
        
        mock_stan_build.assert_called_once()
    
    def test_build_model_runtime_error(self, mock_stan_build, service):
        """Testa build_model com RuntimeError"""
        # Arrange
        mock_stan_build.side_effect = RuntimeError("Stan runtime error")
        
        problematic_data = {
            "N": 0,  # Dados inválidos
            "y": []
        }
        
        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            service.build_model(problematic_data)
        
        # Verifica mensagem de erro personalizada
        expected_msg = "Try specifying initial values, reducing ranges of constrained values, reparameterizing the model or reducing the samples amount."
        assert str(exc_info.value) == expected_msg
        mock_stan_build.assert_called_once()
    
    def test_build_model_other_exception_with_print(self, mock_stan_build, service, capsys):
        """Testa build_model com outras exceções (imprime erro)"""
        # Arrange
        mock_stan_build.side_effect = ValueError("Some other error")
        
        test_data = {"valid": "data"}
        
        # Act
        result = service.build_model(test_data)
        
        # Assert
        captured = capsys.readouterr()
        
        # Deve imprimir tipo e mensagem do erro
        assert "ValueError" in captured.out or "Error:" in captured.out
        assert "Some other error" in captured.out
        
        # Deve retornar None (implicitamente)
        assert result is None
        
        mock_stan_build.assert_called_once()
    
    def test_build_model_system_exit(self, mock_stan_build, service, capsys):
        """Testa build_model com SystemExit (não deve ser capturado)"""
        # Arrange
        mock_stan_build.side_effect = SystemExit()
        
        test_data = {"test": "data"}
        
        # Act & Assert
        with pytest.raises(SystemExit):
            service.build_model(test_data)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_build_model_multiple_successive_calls(self, mock_stan_build, service, capsys):
        """Testa múltiplas chamadas bem-sucedidas"""
        # Arrange
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()
        
        mock_stan_build.side_effect = [model1, model2, model3]
        
        data1 = {"N": 10}
        data2 = {"N": 20}
        data3 = {"N": 30}
        
        # Act
        result1 = service.build_model(data1)
        result2 = service.build_model(data2)
        result3 = service.build_model(data3)
        
        # Assert
        assert mock_stan_build.call_count == 3
        assert result1 == model1
        assert result2 == model2
        assert result3 == model3

# Testes de integração
class TestBuildModelIntegration:
    """Testes de integração para build_model"""
    
    def test_build_model_in_real_workflow(self):
        """Testa build_model em um fluxo de trabalho mais realista"""
        # Este teste simula como o método seria usado
        print("\n=== Testando fluxo de trabalho completo ===")
        
        # 1. Criação do serviço
        mock_model = Mock()
        mock_model.gen_stan_model.return_value = '''
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
        service = BayesianInferenceService(mock_model.gen_stan_model.return_value)
        
        # 2. Dados de teste
        test_data = {
            "N": 10,
            "values": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1]
        }
        
        # 3. Mock do stan.build
        with patch('stan_circular_inference.service.bayesian_inference.stan.build') as mock_build:
            # Cria um posterior mockado mais realista
            model = Mock()
            
            mock_build.return_value = model
            
            # 4. Chama build_model
            posterior = service.build_model(test_data)
            
            # 5. Verificações
            assert posterior == model
            mock_build.assert_called_once_with(mock_model.gen_stan_model.return_value, data=test_data)
    
    def test_error_handling_in_workflow(self, mock_stan_build, service):
        """Testa tratamento de erro em fluxo de trabalho"""
        # Simula um erro durante build_model
        mock_stan_build.side_effect = TypeError("Invalid data type")
        
        try:
            service.build_model({"invalid": "data"})
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert str(e) == "Wrong parameter format for the model!"