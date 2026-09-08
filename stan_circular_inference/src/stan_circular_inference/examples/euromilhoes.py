def load_data(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def preprocess_data(lines):
    temp = ''
    counter = 0
    for l in lines:
        l = l.strip()
        if l:
            if l[0] in '0123456789':
                temp += l
            if l[0] == 'E':
                if counter <= 0:
                    temp += '\n'
                    counter += 1
                    temp += l
                else:
                    temp += l + '\n'
                    counter = 0
    return temp

def write_data(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)

def format_processed_data(lines):
    numbers_list = []
    stars_list = []
    for l in lines:
        if l.startswith('E'):
            stars_list.extend(extract_numbers(l))
        else:
            numbers_list.extend(extract_numbers(l))
    return numbers_list, stars_list

def extract_numbers(line):
    numbers = []
    counter = 0
    temp = ''
    for c in line:
        if c.isdigit():
            counter += 1
            temp += c
            if counter == 2:
                numbers.append(int(temp))
                counter = 0
                temp = ''

    return numbers

values = format_processed_data(load_data('datasets/processed_data.txt'))
assert max(values[1]) <= 12, "Star bigger than 12"

model = """
data {
  int<lower=1> N;                               // Número de sorteios históricos
  array[N, 5] int<lower=1, upper=50> numbers;   // Histórico de números (Matriz N x 5)
  array[N, 2] int<lower=1, upper=12> stars;     // Histórico de estrelas (Matriz N x 2)
}

transformed data {
  // Criamos contadores para saber quantas vezes cada número/estrela saiu
  array[50] int contagem_numeros = rep_array(0, 50);
  array[12] int contagem_estrelas = rep_array(0, 12);
  
  // Contabiliza a frequência acumulada no histórico
  for (t in 1:N) {
    for (i in 1:5) {
      contagem_numeros[numbers[t, i]] += 1;
    }
    for (j in 1:2) {
      contagem_estrelas[stars[t, j]] += 1;
    }
  }
}

parameters {
  // Bloco obrigatório no Stan, mas vazio porque não estamos a fazer inferência estatística
  real dummy; 
}

model {
  // Distribuição prioritária irrelevante apenas para o Stan correr o compilador
  dummy ~ normal(0, 1); 
}

generated quantities {
  vector[50] penalizacao_numeros;
  vector[12] penalizacao_estrelas;
  
  // Calcula a penalização final de cada número e estrela
  // Quanto MENOR o valor resultante, MENOR foi a penalização (maior a probabilidade atual)
  for (k in 1:50) {
    penalizacao_numeros[k] = contagem_numeros[k] * 0.01;
  }
  
  for (m in 1:12) {
    penalizacao_estrelas[m] = contagem_estrelas[m] * 0.015;
  }
}
"""

from stan_circular_inference.service.bayesian_inference import BayesianInferenceService
import numpy as np
import pandas as pd

service = BayesianInferenceService(model)

N_sorteios = int(len(values[1]) / 2)

model_data = {'N': N_sorteios, 'numbers': np.array(values[0]).reshape(N_sorteios, 5), 'stars': np.array(values[1]).reshape(N_sorteios, 2)}

pos = service.build_model(model_data)
fit = service.get_samples(pos, sample_amount=200000)
df = service.get_statistics(fit)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)