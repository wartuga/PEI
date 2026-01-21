from src.stan_circular_inference.factories.parameters_factory import MeanParameter, VarianceParameter

class Normal(MeanParameter):
    def __init__(self, mu: float = 0, sigma: float = 10):
        self.mu = mu
        self.sigma = sigma
    
    def get_code(self) -> str:
        return f'normal({self.mu}, {self.sigma})'
    
    def __str__(self):
        return f'Normal(mu={self.mu}, sigma={self.sigma})'

class Uniform(MeanParameter):
    def __init__(self, lower: float = -3.14, upper: float = 3.14):
        self.lower = lower
        self.upper = upper
    
    def get_code(self) -> str:
        return f'uniform({self.lower}, {self.upper})'
    
    def __str__(self):
        return f'Uniform(lower={self.lower}, upper={self.upper})'

class Gamma(VarianceParameter):
    def __init__(self, alpha: float = 1, beta: float = 1):
        self.alpha = alpha
        self.beta = beta
    
    def get_code(self) -> str:
        return f'gamma({self.alpha}, {self.beta})'
    
    def __str__(self):
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"

class Exponential(VarianceParameter):
    def __init__(self, lambda_: float = 1):
        self.lambda_ = lambda_
    
    def get_code(self) -> str:
        return f'exponential({self.lambda_})'
    
    def __str__(self):
        return f"Exponential(lambda={self.lambda_})"

# TODO testar
class Bernoulli(MeanParameter):
    """Distribuição Bernoulli para variáveis binárias (0 ou 1)"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probabilidade de sucesso (entre 0 e 1)
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Parâmetro p deve estar entre 0 e 1, recebido: {p}")
        self.p = p
    
    def get_code(self) -> str:
        return f'bernoulli({self.p})'
    
    def __str__(self):
        return f'Bernoulli(p={self.p})'

# TODO testar
class Poisson(MeanParameter):
    """Distribuição Poisson para dados de contagem"""
    
    def __init__(self, lambda_: float = 1.0):
        """
        Args:
            lambda_: Taxa média de ocorrências (deve ser > 0)
        """
        if lambda_ <= 0:
            raise ValueError(f"Lambda deve ser positivo, recebido: {lambda_}")
        self.lambda_ = lambda_
    
    def get_code(self) -> str:
        return f'poisson({self.lambda_})'
    
    def __str__(self):
        return f'Poisson(lambda={self.lambda_})'