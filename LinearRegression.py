# CaloriasModel.py
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "Peso": [60, 70, 80, 60, 70, 80],
    "Minutos": [30, 30, 30, 60, 60, 60],
    "Gasto": [200, 240, 280, 400, 480, 560]  # kcal
}

df = pd.DataFrame(data)
X = df[["Peso", "Minutos"]]
y = df[["Gasto"]]

model = LinearRegression()
model.fit(X, y)

def calculateCalories(peso, minutos):
    result = model.predict([[peso, minutos]])[0][0]
    return round(result, 2)