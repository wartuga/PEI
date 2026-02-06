# tests/service/test_normalize_values.py
import pytest
import numpy as np

class TestNormalizeValues:
    
    def test_basic_wrapping(self, service):
        """Teste básico de wrap"""
        values = [0, 90, 180, 270, 360, 450, 720, -90]
        expected = [0, 90, 180, 270, 0, 90, 0, 270]
        result = service.normalize_values(values, 0, 360)
        assert result == expected
    
    def test_negative_range(self, service):
        """Teste com intervalo negativo-positivo"""
        values = [-180, -90, 0, 90, 180, 270]
        expected = [-180, -90, 0, 90, -180, -90]
        result = service.normalize_values(values, -180, 180)
        assert result == expected
    
    def test_positive_range(self, service):
        """Teste com intervalo positivo"""
        values = [0, 1, 2, 3, 4]
        expected = [0, 1, 0, 1, 0]
        result = service.normalize_values(values, 0, 2)
        assert result == expected
    
    def test_offset_range(self, service):
        """Teste com intervalo não começando em 0"""
        values = [40, 50, 60, 70, 80, 90]
        expected = [40, 50, 40, 50, 40, 50]
        result = service.normalize_values(values, 40, 60)
        assert result == expected
    
    def test_edge_cases(self, service):
        """Teste casos de borda"""
        # Exatamente no máximo
        assert service.normalize_values([360], 0, 360) == [0]
        
        # Logo abaixo do máximo (usando == para float)
        result = service.normalize_values([359.999], 0, 360)
        assert result[0] == 359.999
        
        # Exatamente no mínimo
        assert service.normalize_values([0], 0, 360) == [0]
        
        # Valores negativos que wrappam
        assert service.normalize_values([-1], 0, 360) == [359]
        assert service.normalize_values([-360], 0, 360) == [0]
        assert service.normalize_values([-361], 0, 360) == [359]
    
    def test_decimal_values(self, service):
        """Teste com valores decimais"""
        values = [0.0, 0.5, 1.0, 1.5, 2.0]
        expected = [0.0, 0.5, 0.0, 0.5, 0.0]
        result = service.normalize_values(values, 0.0, 1.0)
        assert result == expected
    
    def test_single_value(self, service):
        """Teste com lista de um elemento"""
        assert service.normalize_values([370], 0, 360) == [10]
        assert service.normalize_values([-10], 0, 360) == [350]
        assert service.normalize_values([410], 40, 60) == [50]
        assert service.normalize_values([1.5], 0, 1) == [0.5]
    
    def test_empty_list(self, service):
        """Teste com lista vazia"""
        assert service.normalize_values([], 0, 360) == []
        assert service.normalize_values([], -180, 180) == []
        assert service.normalize_values([], 40, 60) == []
    
    def test_pi_range(self, service):
        """Teste com intervalo -pi a pi"""
        values = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, 3*np.pi/2]
        expected = [-np.pi, -np.pi/2, 0, np.pi/2, -np.pi, -np.pi/2]
        result = service.normalize_values(values, -np.pi, np.pi)
        # Aqui talvez precisemos de approx devido a pi ser irracional
        # Mas na prática, math.pi é uma constante de ponto flutuante
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10  # Tolerância muito pequena
    
    def test_large_values(self, service):
        """Teste com valores muito grandes"""
        # 360*x+10 => x=4,6,9
        values = [1450, 2170, 3250]
        expected = [10, 10, 10]  # 1000 % 360 = 280
        result = service.normalize_values(values, 0, 360)
        assert result == expected
    
    def test_fractional_results(self, service):
        """Teste que produz resultados fracionários"""
        values = [365.5, 730.25, -15.75]
        expected = [5.5, 10.25, 344.25]
        result = service.normalize_values(values, 0, 360)
        assert result == expected
    
    def test_identity_property(self, service):
        """Testa que valores já no intervalo não mudam"""
        values = [45, 90, 135, 180, 225, 270, 315]
        result = service.normalize_values(values, 0, 360)
        assert result == values  # Deve ser exatamente igual
    
    def test_modulo_property(self, service):
        """Testa propriedade matemática do módulo"""
        values = [a * 360 + b for a in range(-3, 4) for b in [0, 90, 180, 270]]
        expected = [b for _ in range(7) for b in [0, 90, 180, 270]]
        result = service.normalize_values(values, 0, 360)
        assert result == expected