from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.factories.parameters_factory import MeanParameter, VarianceParameter
from typing import Dict, Any

class WrappedCauchy(ProbabilisticModel):
    def __init__(self, mu: MeanParameter, rho: VarianceParameter):
        super().__init__("WrappedCauchy")
        self.mu = mu
        self.rho = rho
    
    def gen_stan_model(self) -> str:
        model_code = f"""
            data {{
                int<lower=0> N;
                vector[N] values;
            }}

            parameters {{
                real<lower=-pi(), upper=pi()> mu;
                real<lower=0, upper=1> rho;
            }}

            model {{
                mu ~ {self.mu.get_code()};
                rho ~ {self.rho.get_code()};
                
                for (n in 1:N){{
                    target += log((1.0/(2*pi())) * ((1 - square(rho)) / (1 + square(rho) - 2*rho*cos(values[n] - mu))));
                }}
            }}
            """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': str(self.mu),
            'rho': str(self.rho)
        }