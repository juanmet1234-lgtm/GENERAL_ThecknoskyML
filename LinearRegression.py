import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io, base64

# ---- Modelo gasto energético ----
data_energy = {
    "Peso (kg)": [60, 70, 80, 90, 65, 75, 85, 95, 68, 78, 88, 100],
    "Minutos de ejercicio": [30, 45, 60, 75, 20, 40, 50, 70, 25, 55, 65, 80],
    "Gasto energético (kcal)": [200, 320, 450, 600, 150, 280, 420, 580, 180, 400, 500, 700]
}

df_energy = pd.DataFrame(data_energy)

X_energy = df_energy[["Peso (kg)", "Minutos de ejercicio"]]
y_energy = df_energy[["Gasto energético (kcal)"]]

model_energy = LinearRegression()
model_energy.fit(X_energy, y_energy)

def calculateEnergy(peso, minutos):
    result = model_energy.predict([[peso, minutos]])[0][0]
    return round(result, 2)

def generate_energy_plot(peso=None, minutos=None):
    X1 = df_energy["Peso (kg)"]
    X2 = df_energy["Minutos de ejercicio"]
    y = df_energy["Gasto energético (kcal)"]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    # Puntos de entrenamiento
    ax.scatter(X1, X2, y, color="blue", label="Datos de entrenamiento")

    # Plano de regresión
    x1_range = np.linspace(X1.min(), X1.max(), 20)
    x2_range = np.linspace(X2.min(), X2.max(), 20)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_pred_grid = model_energy.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()])
    y_pred_grid = y_pred_grid.reshape(x1_grid.shape)

    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color="red", alpha=0.5)

    # Si el usuario ingresó valores → dibujar el punto
    if peso is not None and minutos is not None:
        y_pred = model_energy.predict([[peso, minutos]])[0][0]
        ax.scatter(peso, minutos, y_pred, color="green", s=100, label="Tu predicción")

    ax.set_xlabel("Peso (kg)")
    ax.set_ylabel("Minutos de ejercicio")
    ax.set_zlabel("Gasto energético (kcal)")
    ax.set_title("Regresión lineal múltiple - Gasto energético")
    ax.legend()

    # Convertir a base64 para mostrar en HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

