from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.factories.parameters_factory import MeanParameter, VarianceParameter
from typing import Dict, Any

class VonMisesMK(ProbabilisticModel):
    def __init__(self, mu: MeanParameter, kappa: VarianceParameter):
        super().__init__("Von Mises")
        self.mu = mu
        self.kappa = kappa
    
    def gen_stan_model(self) -> str:
        model_code = f"""
            data {{
                int<lower=0> N;
                vector[N] values;
            }}

            parameters {{
                real<lower=0, upper=2*pi()> mu;
                real<lower=0> kappa;
            }}

            model {{
                mu ~ {self.mu.get_code()};
                kappa ~ {self.kappa.get_code()};
                
                values ~ von_mises(mu, kappa);
            }}
        """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': str(self.mu),
            'kappa': str(self.kappa)
        }
    
class VonMisesM(ProbabilisticModel):
    def __init__(self, mu: MeanParameter):
        super().__init__("Von Mises")
        self.mu = mu
    
    def gen_stan_model(self) -> str:
        model_code = f"""
            data {{
                int<lower=1> N;
                array[N] real values;
                real<lower=0> kappa;
            }}
            parameters {{
                real<lower=0, upper=2*pi()> mu;
            }}
            model {{
                real normalizer = log(2 * pi() * modified_bessel_first_kind(0, kappa));
            
                for (n in 1:N) {{
                    target += kappa * cos(values[n] - mu) - normalizer;
                }}
                
                mu ~ uniform(0, 2*pi());
            }}
        """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': str(self.mu)
        }