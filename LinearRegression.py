import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LinearRegression

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

def plot_regression():
    # Gráfico de dispersión
    plt.figure(figsize=(6,4))
    plt.scatter(df_energy["Minutos de ejercicio"], df_energy["Gasto energético (kcal)"], color="blue", label="Datos")

    # Línea de regresión (para un peso fijo, ej: 70 kg)
    minutos_range = [[m, 70] for m in range(20, 90, 5)]
    y_pred = model_energy.predict(minutos_range)
    minutos_x = [m[0] for m in minutos_range]

    plt.plot(minutos_x, y_pred, color="red", label="Regresión (peso=70kg)")
    plt.xlabel("Minutos de ejercicio")
    plt.ylabel("Gasto energético (kcal)")
    plt.title("Regresión Lineal - Gasto Energético")
    plt.legend()

    # Convertir a imagen base64
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{graph_url}"


