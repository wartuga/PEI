from abc import ABC, abstractmethod
from typing import Dict, Any

class ProbabilisticModel(ABC):
    
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}

    @abstractmethod
    def gen_stan_model(self) -> str:
        """Return Stan code of the model"""
        pass

    @abstractmethod
    def get_parameters_prior(self) -> Dict[str, Any]:
        """Return the parameters distributions"""
        pass

    def __str__(self) -> str:
        return f"Model: {self.name}"