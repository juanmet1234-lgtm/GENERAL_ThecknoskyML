import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------
# Cargar y preparar dataset
# ----------------------------
def load_and_prepare(path: str):
    df = pd.read_csv(path)
    df = pd.get_dummies(df, columns=["Tipo_dispositivo"], drop_first=True)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y

# ----------------------------
# Entrenar modelo
# ----------------------------
def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# ----------------------------
# Evaluar modelo
# ----------------------------
def evaluate(model, X_test, y_test, show_plot=True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No", "Sí"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    img_base64 = None
    if show_plot:
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "cm_image": img_base64
    }

# ----------------------------
# Predicción
# ----------------------------
def predict_label(model, features, threshold=0.5):
    prob = model.predict_proba([features])[0][1]
    label = "Sí" if prob >= threshold else "No"
    return label, prob

