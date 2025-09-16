from flask import Flask, render_template, request
import pandas as pd
from LinearRegression import calculateEnergy, generate_energy_plot
from Churn_Logistic import load_and_prepare, train_model, evaluate, predict_label

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


@app.route('/Energy', methods=['GET', 'POST'])
def Energy():
    energy = None
    plot_url = None
    if request.method == 'POST':
        peso = float(request.form['peso'])
        minutos = float(request.form['minutos'])
        energy = calculateEnergy(peso, minutos)
        plot_url = generate_energy_plot(peso, minutos)

    return render_template('Energy.html', energy=energy, plot_url=plot_url)


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


if __name__ == '__main__':
    app.run(debug=True)
