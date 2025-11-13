import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("txtdata.csv")
n = len(data)

# Criar ângulos igualmente espaçados
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

# Criar figura com projeção polar
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# Adicionar números à volta da circunferência
for i, (angle, value) in enumerate(zip(angles, data)):
    if i % 5 == 0:
        # Posicionar o texto ligeiramente fora do ponto mais distante
        label_radius = max(data) * 1.1
        ax.text(angle, label_radius, f"{i+1}", 
                ha='center', va='center', fontsize=8)

# Configurações do gráfico
ax.set_theta_offset(np.pi/2)  # Começar do topo (0° no topo)
ax.set_theta_direction(-1)    # Sentido horário
ax.set_ylim(0, max(data) * 1.2)
ax.grid(True)


min_val = min(data)
width = 2 * np.pi / n
bars = ax.bar(angles, data, width, min_val)

plt.title('Diagrama de Rosas')
plt.show()



# inserir os dados no gráfico de barras depois das simulações