from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.factories.parameters_factory import MeanParameter, VarianceParameter
from typing import Dict, Any

class Cardioid(ProbabilisticModel):
    def __init__(self, mu: MeanParameter, rho: VarianceParameter):
        super().__init__("Cardioid")
        self.mu = mu
        self.rho = rho
    
    def gen_stan_model(self) -> str:
        model_code = f"""
            data {{
                int<lower=0> N;
                vector[N] values;
            }}

            parameters {{
                real<lower=0, upper=2*pi()> mu;
                real<lower=0, upper=0.5> rho;
            }}

            model {{
                mu ~ {self.mu.get_code()};
                rho ~ {self.rho.get_code()};
                
                for (n in 1:N){{
                    target += -log(2*pi()) + log1p(2*rho*cos(values[n] - mu));
                }}
            }}
            """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': self.mu,
            'rho': self.rho
        }