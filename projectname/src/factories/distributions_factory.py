from projectname.src.factories.parameters_factory import MeanParameter, VarianceParameter

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

# TODO
# Poisson
# Bernoulli