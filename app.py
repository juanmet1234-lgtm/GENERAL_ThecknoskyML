from flask import Flask, render_template, request
from LinearRegression import calculateEnergy, generate_energy_plot
from Churn_Logistic import load_and_prepare, train_model, evaluate, predict_label
# añadir plot_trajectory y funciones auxiliares
from reinforcement_agent import GridWorld, QLearningAgent, plot_rewards, plot_policy, create_trajectory_gif, plot_trajectory
import os
import joblib
import pandas as pd

# Variable global
trained_rewards = None
trained_agent = None


# Cargar modelo y métricas de fraude
fraude_data = joblib.load("fraude_model.pkl")
fraude_model = fraude_data["model"]
fraude_accuracy = fraude_data["metrics"]["accuracy"]

app = Flask(__name__)

# ----------------------------
# Cargar datos y entrenar modelo de churn
# ----------------------------
X, y = load_and_prepare("churn_streaming.csv")
model, X_test, y_test = train_model(X, y)
metrics = evaluate(model, X_test, y_test, show_plot=True)

# ----------------------------
# Rutas principales
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/MenuMLcasos')
def MenuMLcasos():
    return render_template('MenuMLcasos.html')


@app.route('/FiltradoDeCorreos')
def Filtrado_de_correos():
    return render_template('FiltradoDeCorreos.html')


@app.route('/FraudesBancarios')
def FraudesBancarios():
    return render_template('FraudesBancarios.html')


@app.route('/PagLugo')
def pag_lugo():
    return render_template('PagLugo.html')


@app.route('/PagCamero')
def pag_camero():
    return render_template('PagCamero.html')


@app.route('/MunuLR')
def Menu_LR():
    return render_template('MenuLR.html')


@app.route('/MunuReLo')
def Menurelo():
    return render_template('MenuReLo.html')




@app.route('/LR', methods=['GET', 'POST'])
def Energy():
    energy = None
    plot_url = None
    if request.method == 'POST':
        peso = float(request.form['peso'])
        minutos = float(request.form['minutos'])
        energy = calculateEnergy(peso, minutos)
        plot_url = generate_energy_plot(peso, minutos)

    return render_template('LR.html', energy=energy, plot_url=plot_url)


@app.route('/Investigacion')
def Investigacion():
    return render_template('Investigacion.html')


# ----------------------------
# A4_practica
# ----------------------------
@app.route('/A4_practica', methods=['GET', 'POST'])
def A4_practica():
    energy = None
    plot_url = None
    if request.method == 'POST':
        peso = float(request.form['peso'])
        minutos = float(request.form['minutos'])
        energy = calculateEnergy(peso, minutos)
        plot_url = generate_energy_plot(peso, minutos)

    return render_template('A4_practica.html', energy=energy, plot_url=plot_url)


# ----------------------------
# A5_practica (predicción de churn)
# ----------------------------
@app.route('/A5_practica', methods=['GET', 'POST'])
def A5_practica():
    pred_result = None
    prob = None

    if request.method == 'POST':
        horas = int(request.form['horas'])
        meses = int(request.form['meses'])
        perfiles = int(request.form['perfiles'])
        dispositivo = request.form['dispositivo']

        # One-hot encoding manual
        disp_PC = 1 if dispositivo == "PC" else 0
        disp_SmartTV = 1 if dispositivo == "SmartTV" else 0
        features = [horas, meses, perfiles, disp_PC, disp_SmartTV]

        # Predicción
        pred_result, prob = predict_label(model, features)

    # Generar reporte HTML dentro de la función para evitar NameError
    report_df = pd.DataFrame(metrics["report"]).transpose()
    report_html = report_df.to_html(classes="table table-striped", float_format="{:.2f}".format)

    return render_template(
        "A5_practica.html",
        metrics=metrics,
        report_html=report_html,
        pred_result=pred_result,
        prob=prob
    )
    
    
@app.route('/Menu_TiposClasificacion')
def Menu_TiposClasificacion():
    return render_template('Menu_TiposClasificacion.html')

@app.route('/A7_Investigaciones')
def A7Inv():
    return render_template('A7_Investigaciones.html')

@app.route('/A12_ReinforcementLearning')
def A12_ReinforcementLearning():
    return render_template('A12_ReinforcementLearning.html')

@app.route('/conceptos_reinforcement_learning')
def conceptos_reinforcement_learning():
    return render_template('conceptos_reinforcement_learning.html')

    

# ----------------------------
# A7_practica - Detección de fraude
# ----------------------------
@app.route('/A7_practica', methods=['GET', 'POST'])
def A7_practica():
    prediction = None
    probability = None
    interpretation = None

    if request.method == 'POST':
        # Obtener datos del formulario
        monto = float(request.form['monto'])
        hora = int(request.form['hora'])
        ubicacion = request.form['ubicacion']
        dispositivo = request.form['dispositivo']
        historial = int(request.form['historial'])
        threshold = float(request.form['threshold']) if request.form['threshold'] else 0.5

        # Crear DataFrame con una fila
        input_data = pd.DataFrame([{
            "Monto": monto,
            "Hora": hora,
            "Ubicación": ubicacion,
            "Dispositivo": dispositivo,
            "Historial": historial
        }])

        # Predecir directamente usando el pipeline entrenado
        prob = fraude_model.predict_proba(input_data)[:, 1][0]


        # Aplicar threshold
        pred = "Sí" if prob >= threshold else "No"

        # Interpretación del threshold
        if threshold < 0.5:
            interpretation = f"Con threshold={threshold} la sensibilidad aumenta, pero también los falsos positivos."
        elif threshold > 0.5:
            interpretation = f"Con threshold={threshold} la precisión aumenta, pero puedes perder algunos fraudes."
        else:
            interpretation = "Con threshold=0.5 se mantiene el balance entre sensibilidad y precisión."

        prediction = pred
        probability = round(prob, 4)

    return render_template(
        'A7_practica.html',
        prediction=prediction,
        probability=probability,
        interpretation=interpretation,
        fraude_accuracy=round(fraude_accuracy, 4)
    )
    
@app.route('/practica_reinforcement_learning', methods=['GET', 'POST'])
def practica_reinforcement_learning():
    rewards_plot = None
    policy_text = None
    qtable_text = None
    message = None

    if request.method == 'POST':

        # Leer parámetros
        episodes = int(request.form.get('episodes', 200))
        alpha = float(request.form.get('alpha', 0.1))
        gamma = float(request.form.get('gamma', 0.9))
        epsilon = float(request.form.get('epsilon', 0.2))
        grid_size = 4 

        # Crear entorno
        env = GridWorld(size=grid_size)

        # Crear agente
        agent = QLearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)

        # Entrenamiento
        rewards = agent.train(episodes=episodes)

        # ---------- 1) GRAFICA DE RECOMPENSAS ----------
        try:
            rewards_plot = plot_rewards(rewards)
        except Exception as e:
            message = f"Error generando gráfica: {e}"

        # ---------- 2) Q-TABLE COMO TEXTO ----------
        try:
            qtable_text = str(agent.q_table)
        except:
            qtable_text = "No se pudo generar la Q-Table."

        # ---------- 3) POLÍTICA APRENDIDA COMO TEXTO ----------
        try:
            policy_matrix = agent.get_policy()  
            policy_text = str(policy_matrix)
        except:
            policy_text = "No se pudo generar la política aprendida."

    return render_template(
        'practica_reinforcement_learning.html',
        rewards_plot=rewards_plot,
        policy=policy_text,
        qtable=qtable_text,
        message=message
    )

if __name__ == '__main__':
    app.run(debug=True)
