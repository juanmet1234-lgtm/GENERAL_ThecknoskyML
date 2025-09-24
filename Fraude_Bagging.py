import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------------------
# 1. Cargar dataset
# -------------------------------
df = pd.read_csv("fraude_dataset.csv")

# -------------------------------
# 2. Definir variables
# -------------------------------
X = df.drop("Fraude", axis=1)
y = df["Fraude"]

# Variables numéricas (requieren escalado)
num_features = ["Monto", "Hora", "Historial"]

# Variables categóricas (requieren One-Hot)
cat_features = ["Ubicación", "Dispositivo"]

# -------------------------------
# 3. Preprocesamiento
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# -------------------------------
# 4. Dividir en train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Definir modelo Bagging
# -------------------------------
base_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", BaggingClassifier(
        estimator=base_clf,
        n_estimators=50,
        random_state=42
    ))
])

# -------------------------------
# 6. Entrenamiento
# -------------------------------
model.fit(X_train, y_train)


# -------------------------------
# 7. Evaluación
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n Exactitud:", accuracy)

# Reporte de clasificación
report = classification_report(y_test, y_pred, target_names=["Legítima (0)", "Fraudulenta (1)"])
print("\n Reporte de Clasificación:\n", report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred: 0 (Legítima)", "Pred: 1 (Fraude)"],
            yticklabels=["Real: 0 (Legítima)", "Real: 1 (Fraude)"])
plt.title("Matriz de Confusión")
plt.savefig("static/imagenes/CM_Fraudes.png")
plt.close()
print(" Matriz de confusión guardada como confusion_matrix.png")

# -------------------------------
# 8. Guardar modelo entrenado
# -------------------------------

metrics_fraude = {
    "accuracy": accuracy,
    "classification_report": report
}

joblib.dump({
    "model": model,
    "metrics": metrics_fraude
}, "fraude_model.pkl")
print(" Modelo y métricas guardados en fraude_model.pkl")