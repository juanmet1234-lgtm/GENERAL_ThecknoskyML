import pandas as pd
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
