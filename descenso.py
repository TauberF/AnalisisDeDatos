#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
df_datos = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Boston.csv')
y = df_datos['MEDV']
X = df_datos.drop('MEDV', axis=1)
# %% Dividir datos en entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#%%
# Función de costo MSE
def compute_cost(X, y, beta):
    m = len(y)
    J = np.sum((X.dot(beta) - y) ** 2) / (2 * m)
    return J

# Gradiente descendente
def gradient_descent(X, y, beta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        beta = beta - (alpha / m) * X.T.dot(X.dot(beta) - y)
        cost_history[i] = compute_cost(X, y, beta)
                
    return beta, cost_history

#%%
# Inicialización
alpha = 0.00000001
iterations = 10000
beta = np.zeros(X_train.shape[1])

# Ejecutar el gradiente descendente
beta, cost_history = gradient_descent(X_train, y_train, beta, alpha, iterations)

# %% Graficar la función de costo
plt.plot(range(1, iterations + 1), cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Iteraciones')
plt.ylabel('Costo')
plt.title('Descenso del gradiente')
plt.show()


# %%
beta
# %%
