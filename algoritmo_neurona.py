import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def funcion_activacion(x):
    return np.where(x >= 0, 1, 0)

def seleccionar_archivo_csv():
    root = tk.Tk()
    root.withdraw()  # Ocultamos la ventana principal de Tkinter
    archivo_csv = filedialog.askopenfilename(title="Seleccionar archivo CSV",
                                             filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")))
    root.destroy()
    return archivo_csv

def obtener_parametros():
    root = tk.Tk()
    root.withdraw()
    eta = simpledialog.askfloat("Tasa de Aprendizaje", "Ingrese la tasa de aprendizaje:",
                                minvalue=0.0, maxvalue=1.0, parent=root)
    epocas = simpledialog.askinteger("Número de Iteraciones", "Ingrese el número de iteraciones:",
                                      minvalue=1, parent=root)
    root.destroy()
    return eta, epocas

archivo_csv = seleccionar_archivo_csv()
if not archivo_csv:
    sys.exit("No se seleccionó ningún archivo CSV.")

df = pd.read_csv(archivo_csv, delimiter='[,;]', engine='python')

# Separar el DataFrame en X y Y
df_x = df.iloc[:, :-1]
df_y = df.iloc[:, -1]
df_x.insert(0, 'x0', 1)
num_columnas = df_x.shape[1]

X = df_x.to_numpy()
Yd = df_y.to_numpy()
W = np.round(np.random.uniform(-1, 1, (num_columnas, 1)), 1)

eta, epocas = obtener_parametros()
tasas_error = []
historial_pesos = []

for epoca in range(epocas):
    U = np.linalg.multi_dot([X, W])
    Yc = funcion_activacion(U)
    E = Yd.reshape(-1, 1) - Yc
    historial_pesos.append(W.copy())
    ΔW = eta * np.linalg.multi_dot([X.T, E])
    norma_error = np.linalg.norm(E)
    W += ΔW
    tasas_error.append(norma_error)

# Gráfico de la evolución del error
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(tasas_error) + 1), tasas_error, marker='o')
plt.title('Evolución de la Norma del Error por Época')
plt.xlabel('Época')
plt.ylabel('Norma del Error')
plt.grid(True)

# Gráfico de la evolución de los pesos
pesos = np.array(historial_pesos)
plt.subplot(1, 2, 2)
for i in range(pesos.shape[1]):
    plt.plot(range(1, len(historial_pesos) + 1), pesos[:, i, 0], label=f'Peso {i}')

plt.title('Evolución de los Pesos por Época')
plt.xlabel('Época')
plt.ylabel('Valor del Peso')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

def mostrar_info():
    pesos_iniciales = historial_pesos[0]
    pesos_finales = historial_pesos[-1]
    info = f"Configuración de pesos inicial: {pesos_iniciales}\n\n"
    info += f"Configuración de pesos final: {pesos_finales}\n\n"
    info += f"Tasa de aprendizaje: {eta}\n"
    info += f"Número de iteraciones: {len(historial_pesos)}"

    messagebox.showinfo("Información del Entrenamiento", info)

mostrar_info()