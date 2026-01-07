from abc import ABC, abstractmethod

class MeanParameter(ABC):
    @abstractmethod
    def get_code(self) -> str:
        pass

class VarianceParameter(ABC):
    @abstractmethod
    def get_code(self) -> str:
        pass