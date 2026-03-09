from stan_circular_inference.service.bayesian_inference import BayesianInferenceService
from stan_circular_inference.service.data_type import DataType
import numpy as np
from scipy.stats import vonmises

model = """
data {
    int<lower=0> N;
    vector[N] values;
    // Mixing weight for the first component (between 0 and 1)
    // real<lower=0, upper=1> mixing_weight;
}

parameters {
    real<lower=0, upper=2*pi()> mu1;
    real<lower=3*pi()/2, upper=2*pi()> mu2;
    
    real<lower=0> kappa1;
    real<lower=0> kappa2;
    
    // Optional: If you want to estimate the mixing weight and remove it from data
    // real<lower=0, upper=1> mixing_weight;
    // vetor de pontos dados em vez de um valor real
    vector<lower=0, upper=1>[N] mixing_weight;
}

transformed parameters {
    // real<lower=0, upper=2*pi()> mu;

    // Compute circular mean
    // mu = atan2(sin(mu1) + sin(mu2), cos(mu1) + cos(mu2));
    
    // Adjust to [0, 2π) if needed
    // if (mu < 0) {
    //     mu = mu + 2 * pi();
    // }
}

model {
    // Priors for the parameters
    mu1 ~ normal(0, pi()/2);
    mu2 ~ uniform(3*pi()/2, 2*pi());
    kappa1 ~ exponential(0.1);
    kappa2 ~ exponential(0.1);
    mixing_weight ~ beta(1, 1);
    
    // Mixture model likelihood using log_sum_exp (it prevents underflow and overflow)
    for (n in 1:N) {
        target += log_sum_exp(
            log(mixing_weight[n]) + von_mises_lpdf(values[n] | mu1, kappa1),
            log(1 - mixing_weight[n]) + von_mises_lpdf(values[n] | mu2, kappa2)
        );
    }
}
"""

mu1 = np.pi/4
mu2 = -np.pi/4
kappa = 9
size = 100

samples1 = vonmises.rvs(kappa, loc=mu1, size=size)
samples2 = vonmises.rvs(kappa, loc=mu2, size=size)

# pensar em normalizar a samples2 para coincidir com a dist2

samples = np.concatenate([samples1, samples2])
#np.random.shuffle(samples)

service = BayesianInferenceService(model)

data = {'N': len(samples), 'values': samples}

posterior = service.build_model(data=data)
fit = service.get_samples(posterior=posterior, sample_amount=1000)

dist1 = [
    1 if sum(1 for v in values if v > 0.5) >= len(values) / 2 else 0
    for values in fit['mixing_weight']
]

real_dist1 = [1] * size
real_dist2 = [0] * size

real_values = real_dist1 + real_dist2

incorrect_values = [(real_val, val) for real_val, val in zip(real_values, dist1) if real_val != val]

values = service.get_values(fit=fit, 
        parameters=[
            'mu1', 'mu2', 
            'mixing_weight.1',
            'mixing_weight.2',
            'mixing_weight.3',
            'mixing_weight.4',
            'mixing_weight.5',
            'mixing_weight.6',
            'mixing_weight.7',
            'mixing_weight.8',
            'mixing_weight.9',
            'mixing_weight.10',
            'mixing_weight.11',
            'mixing_weight.12',
            'mixing_weight.13',
            'mixing_weight.14',
            'mixing_weight.15',
            'mixing_weight.16',
            'mixing_weight.17',
            'mixing_weight.18',
            'mixing_weight.19',
            'mixing_weight.20',
            'mixing_weight.195'
        ])

service.multiple_graphics(
    values,
    min_val=0, 
    max_val=2*np.pi, 
    param_names=[
        'mu1', 'mu2',
        'mixing_weight.1',
        'mixing_weight.2',
        'mixing_weight.3',
        'mixing_weight.4',
        'mixing_weight.5',
        'mixing_weight.6',
        'mixing_weight.7',
        'mixing_weight.8',
        'mixing_weight.9',
        'mixing_weight.10',
        'mixing_weight.11',
        'mixing_weight.12',
        'mixing_weight.13',
        'mixing_weight.14',
        'mixing_weight.15',
        'mixing_weight.16',
        'mixing_weight.17',
        'mixing_weight.18',
        'mixing_weight.19',
        'mixing_weight.20', 
        'mixing_weight.195'
    ],
    parameters_type=[True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])

# statistics = service.match_points_to_distributions(samples, real_values, dist1, min_val=0, max_val=2*np.pi, data_type=DataType.RADS)

# print(statistics['confusion_matrix'])

# service.get_statistics(fit)

#values = service.get_pystan_statistics(data=data, parameters=['mixing_weight.1'], sample_amount=100)
# service.circular_graphic(values['mixing_weight.1'], min_val=0, max_val=1)
#service.circular_graphic(values['mu2'], min_val=0, max_val=2*np.pi)