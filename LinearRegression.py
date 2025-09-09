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

# ---- Modelo notas ----
data_grade = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}
df_grade = pd.DataFrame(data_grade)
X_grade = df_grade[["Study Hours"]]
y_grade = df_grade[["Final Grade"]]
model_grade = LinearRegression()
model_grade.fit(X_grade, y_grade)

def calculateGrade(hours):
    result = model_grade.predict([[hours]])[0][0]
    return round(result, 2)
