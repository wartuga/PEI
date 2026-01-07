# TODO to include in the final document as example

# Random values from ChatGPT just to test the model

import numpy as np
import time
import utils
import matplotlib.pyplot as plt

model = """
data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}

parameters {
  real a;
  real b;
  real<lower=0> sigma;
}

model {
  a ~ normal(0, 10);
  b ~ normal(0, 10);

  y ~ normal(a * x + b, sigma);
}
"""

# Parâmetros reais para geração de dados
altura_media = 170
peso_base = 70
inclinacao_real = 0.6  # 0.6 kg por cm
intercepto_real = -32  # peso quando altura = 0
desvio = 5

length_data = 100

# Gerar dados de forma realista
heights = np.random.normal(altura_media, 10, length_data)
weights = inclinacao_real * heights + intercepto_real + np.random.normal(0, desvio, length_data)

model_data = {'N': length_data, 'heights': heights, 'weights': weights}

print("Gerando dados...")
print(f"Alturas: média = {np.mean(heights):.1f} cm, desvio = {np.std(heights):.1f} cm")
print(f"Pesos: média = {np.mean(weights):.1f} kg, desvio = {np.std(weights):.1f} kg")

start_time = time.time()

# Estimar os parâmetros
interest_parameter_values_a = utils.get_pystan_statistics(
    model_data=model_data, 
    model=model, 
    parameter='a', 
    sample_amount=5000
)

interest_parameter_values_b = utils.get_pystan_statistics(
    model_data=model_data, 
    model=model, 
    parameter='b', 
    sample_amount=5000
)

print(f"\nTempo de execução: {time.time() - start_time:.2f} segundos")

# Resultados
print("\n=== RESULTADOS DA REGRESSÃO ===")
print(f"Parâmetro 'a' (inclinação):")
print(f"  Real: {inclinacao_real:.3f}")
print(f"  Estimado: {interest_parameter_values_a.mean():.3f}")

print(f"\nParâmetro 'b' (intercepto):")
print(f"  Real: {intercepto_real:.3f}")
print(f"  Estimado: {interest_parameter_values_b.mean():.3f}")

# Gráfico de dispersão com a reta ajustada
plt.figure(figsize=(10, 6))
plt.scatter(heights, weights, alpha=0.7, label='Dados observados')

# Reta real (se conhecida)
x_range = np.linspace(np.min(heights), np.max(heights), 100)
y_real = inclinacao_real * x_range + intercepto_real
plt.plot(x_range, y_real, 'g--', linewidth=2, label=f'Reta real: y = {inclinacao_real:.2f}x + {intercepto_real:.2f}')

# Reta estimada
y_estimado = interest_parameter_values_a['mean'] * x_range + interest_parameter_values_b['mean']
plt.plot(x_range, y_estimado, 'r-', linewidth=2, label=f'Reta estimada: y = {interest_parameter_values_a["mean"]:.2f}x + {interest_parameter_values_b["mean"]:.2f}')

plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.title('Regressão Linear: Peso vs Altura')
plt.grid(True, alpha=0.3)
plt.show()