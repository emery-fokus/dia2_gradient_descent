import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Chargement du dataset
df = pd.read_csv('prix_maisons.csv')

# 2. Normalisation
x_means = df['surface'].mean()
x_std = df['surface'].std()
y_means = df['prix'].mean()
y_std = df['prix'].std()

df['surface'] = (df['surface'] - x_means) / x_std
df['prix'] = (df['prix'] - y_means) / y_std
plt.scatter(df['surface'], df['prix'])
plt.xlabel('Surface (normalisée)')
plt.ylabel('Prix (normalisé)')
plt.title('Données normalisées')
plt.show()
# Extraction des valeurs pour le calcul 
x = df['surface'].values
y = df['prix'].values

# 3. Fonctions du modèle
def quadratic_regression(a, b, c, x):
    return a * x**2 + b * x + c

def mse(y_pre, y):
    return np.mean((y_pre - y)**2)

def rmse(y_pre, y):
    return np.sqrt(mse(y_pre, y))

def backpropagation(a, b, c, x, y, lr):
    y_pred = a * x**2 + b * x + c
    error = y_pred - y
    
    # Gradients 
    dl_da = np.mean(2 * error * x**2)
    dl_db = np.mean(2 * error * x)
    dl_dc = np.mean(2 * error) 

    a = a - lr * dl_da
    b = b - lr * dl_db
    c = c - lr * dl_dc
    
    rmse_value = np.sqrt(np.mean(error**2))
    return a, b, c, rmse_value, y_pred

def gradient_descent_quadratics(x, y, epochs=1000, lr=0.01):
    a, b, c = np.random.rand(), np.random.rand(), np.random.rand()
    rmse_history = []
    y_pred_final = None # Variable pour stocker la dernière prédiction
    
    for i in range(epochs):
        a, b, c, rmse_val, y_pred_final = backpropagation(a, b, c, x, y, lr)
        rmse_history.append(rmse_val)
    
    return a, b, c, rmse_history, y_pred_final

#  Lancement de l'entraînement
a_final, b_final, c_final, rmse_history, y_pred = gradient_descent_quadratics(x, y)

# 5. Visualisation 
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label="Données")
# On trie x pour que le tracé de la ligne soit propre
sorted_indices = np.argsort(x)
plt.plot(x[sorted_indices], y_pred[sorted_indices], color="red", label="Prédiction")
plt.legend()
plt.title("Ajustement du modèle")
plt.show()

#  Graphique de la RMSE 
plt.plot(rmse_history)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Training Progress')
plt.show()