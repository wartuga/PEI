from stan_circular_inference.factories.model_factory import ProbabilisticModel
from stan_circular_inference.factories.parameters_factory import MeanParameter, VarianceParameter
from typing import Dict, Any

class Mixture(ProbabilisticModel):
    def __init__(self, dist1: ProbabilisticModel, dist2: ProbabilisticModel):
        super().__init__(f"Mixture_{dist1.name}_{dist2.name}")
        self.dist1 = dist1
        priors1 = dist1.get_parameters_prior()
        if self.dist1.name == 'Von Mises':
            self.dist1_mu = priors1['mu'].get_code()
            self.dist1_kappa = priors1['kappa'].get_code()
        else:
            self.dist1_mu = priors1['mu'].get_code()
            self.dist1_rho = priors1['rho'].get_code()
        self.dist2 = dist2
        priors2 = dist2.get_parameters_prior()
        if self.dist2.name == 'Von Mises':
            self.dist2_mu = priors2['mu'].get_code()
            self.dist2_kappa = priors2['kappa'].get_code()
        else:
            self.dist2_mu = priors2['mu'].get_code()
            self.dist2_rho = priors2['rho'].get_code()

    # vonmises: von_mises_lpdf(values[n] | mu1, kappa1)
    # wrappedcauchy: target += -log(2*pi()) + log1m(rho2) - log1p(rho2 - 2*rho*cos(values[n] - mu));
    # cardioid: target += -log(2*pi()) + log1p(2*rho*cos(values[n] - mu));
    
    def gen_stan_model(self) -> str:
        model_code = f"""

            {'' if self.dist1.name == 'Von Mises' and self.dist2.name == 'Von Mises'
            else '''
            functions{
                real cardioid_lpdf(real value, real mu, real rho) {
                    return -log(2*pi()) + log1p(2*rho*cos(value - mu));
                }

                real wrapped_cauchy_lpdf(real value, real mu, real rho) {
                    real rho2 = square(rho);
                    return -log(2*pi()) + log1m(rho2) - log1p(rho2 - 2*rho*cos(value - mu));
                }
            }
            ''' if self.dist1.name == 'Cardioid' and self.dist2.name == 'Wrapped Cauchy' or self.dist1.name == 'Wrapped Cauchy' and self.dist2.name == 'Cardioid'
            else '''
            functions{
                real cardioid_lpdf(real value, real mu, real rho) {
                    return -log(2*pi()) + log1p(2*rho*cos(value - mu));
                }
            }
            ''' if self.dist1.name == 'Cardioid' or self.dist2.name == 'Cardioid'
            else '''
            functions{
                real wrapped_cauchy_lpdf(real value, real mu, real rho) {
                    real rho_sqr = square(rho);
                    return -log(2*pi()) + log1m(rho_sqr) - log1p(rho_sqr - 2*rho*cos(value - mu));
                }
            }
            '''}

            data {{
                int<lower=0> N;
                vector[N] values;
            }}

            parameters {{
                real<lower=0, upper=2*pi()> mu1;
                real<lower=0, upper=2*pi()> mu2;
                
                {'real<lower=0> kappa1;' if self.dist1.name == 'Von Mises' 
                else 'real<lower=0, upper=0.5> rho1;' if self.dist1.name == 'Cardioid'
                else 'real<lower=0, upper=1> rho1;'}
                {'real<lower=0> kappa2;' if self.dist2.name == 'Von Mises'
                else 'real<lower=0, upper=0.5> rho2;' if self.dist2.name == 'Cardioid'
                else 'real<lower=0, upper=1> rho2;'}

                vector<lower=0, upper=1>[N] mixing_weight;
            }}

            model {{
                mixing_weight ~ beta(1, 1);
                mu1 ~ {self.dist1_mu};
                {f'kappa1 ~ {self.dist1_kappa}' if self.dist1.name == 'Von Mises' else f'rho1 ~ {self.dist1_rho}'};

                mu2 ~ {self.dist1_mu};
                {f'kappa2 ~ {self.dist2_kappa}' if self.dist2.name == 'Von Mises' else f'rho2 ~ {self.dist2_rho}'};
                
                // Mixture model likelihood using log_sum_exp (it prevents underflow and overflow)
                for (n in 1:N) {{
                    target += log_sum_exp(
                        log(mixing_weight[n]) + {'von_mises_lpdf(values[n] | mu1, kappa1)' if self.dist1.name == 'Von Mises'
                                                 else 'cardioid_lpdf(values[n] | mu1, rho1)' if self.dist1.name == 'Cardioid'
                                                 else 'wrapped_cauchy_lpdf(values[n] | mu1, rho1)'},
                        log(1 - mixing_weight[n]) + {'von_mises_lpdf(values[n] | mu2, kappa2)' if self.dist2.name == 'Von Mises'
                                                 else 'cardioid_lpdf(values[n] | mu2, rho2)' if self.dist2.name == 'Cardioid'
                                                 else 'wrapped_cauchy_lpdf(values[n] | mu2, rho2)'}
                    );
                }}
            }}
            """

        return model_code
    
    def get_parameters_prior(self) -> Dict[str, Any]:
        return {
            'mu': self.mu,
            'rho': self.rho
        }
    
    def __cardioid_function_code():
        return f"""
        real cardioid_lpdf(real value, real mu, real rho) {{
            return -log(2*pi()) + log1p(2*rho*cos(value - mu));
        }}
        """
    
    def __wrapped_cauchy_function_code():
        return f"""
        real wrapped_cauchy_lpdf(real value, real mu, real rho) {{
            real rho2 = square(rho);
            return -log(2*pi()) + log1m(rho2) - log1p(rho2 - 2*rho*cos(value - mu));
        }}
        """