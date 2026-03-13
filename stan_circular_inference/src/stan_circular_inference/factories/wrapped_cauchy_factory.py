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
            //functions {{
            //    real wrapped_cauchy_lpdf(vector y, real mu, real rho) {{
            //        int N = rows(y);
            //        vector[N] log_dens;
            //        for (i in 1:N) {{
            //            log_dens[i] = log(1 - square(rho)) - log(2 * pi() * (1 + square(rho) - 2 * rho * cos(y[i] - mu)));
            //        }}
            //        return sum(log_dens);
            //    }}
            //}}

            data {{
                int<lower=0> N;
                vector[N] values;
            }}

            parameters {{
                real<lower=0, upper=2*pi()> mu; // real<lower=0, upper=2*pi()> mu;
                real<lower=0, upper=1> rho;
            }}

            model {{
                mu ~ {self.mu.get_code()};
                rho ~ {self.rho.get_code()};

                real rho2 = square(rho);

                //values ~ wrapped_cauchy(mu, rho);

                //target += N * (-log(2*pi()) + log1m(rho2)) - sum(log1p(rho2 - 2*rho*cos(values - mu)));
                
                for (n in 1:N){{
                    // phi[i] <- -log( (1-pow(rho,2)) / (2*pi*(1+pow(rho,2)-2*rho*cos(x[i]-mu))) ) + C => -log((1-square_rho) / (2*pi*(1+square_rho-2*rho*cos(values[n] - mu))));
                    // log((1.0/(2*pi)) * ((1 - square_rho) / (1 + square_rho - 2*rho*cos(values[n] - mu))));
                    target += -log(2*pi()) + log1m(rho2) - log1p(rho2 - 2*rho*cos(values[n] - mu));
                }}
            }}
            """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': str(self.mu),
            'rho': str(self.rho)
        }
    
    """
    O modelo do commit demora 4h+ para correr 30k samples
    Testar com este modelo:
    data {
        int<lower=0> N;
        vector[N] values;
    }

    transformed data {
        real const_term = 1.0 / (2 * pi());  // constante uma única vez
        real log_const = log(const_term);
    }

    parameters {
        real<lower=-pi(), upper=pi()> mu;
        real<lower=0, upper=1> rho;
    }

    model {
        // Priors
        mu ~ {self.mu.get_code()};
        rho ~ {self.rho.get_code()};
        
        // Likelihood VETORIZADA (MUITO mais rápida)
        target += N * log_const + 
                N * log1m(square(rho)) - 
                sum(log1p(square(rho) - 2 * rho * cos(values - mu)));
    }
    """