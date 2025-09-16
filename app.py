from flask import Flask, render_template, request  
from LinearRegression import calculateEnergy, generate_energy_plot

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

# TRABAJO 1 ----------------------------------------------------------------------------------
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

# TRABAJO 2 ----------------------------------------------------------------------------------

@app.route('/MunuLR')
def Menu_LR():
    return render_template('MenuLR.html')

@app.route('/MunuReLo')
def Menurelo():
    return render_template('MenuReLo.html')

@app.route('/Energy', methods=['GET', 'POST'])
def Energy():
    return render_template('Energy.html')

@app.route('/Investigacion')
def Investigacion():
    return render_template('Investigacion.html')

@app.route('/LR', methods=['GET', 'POST'])
def Lr():
    calculateResult = None
    plot_url = None  # por defecto no hay gráfico

    if request.method == 'POST':
        peso = float(request.form['peso'])
        minutos = float(request.form['minutos'])
        calculateResult = calculateEnergy(peso, minutos)
        # generar el gráfico con los datos ingresados
        plot_url = generate_energy_plot(peso, minutos)

    return render_template('LR.html', result=calculateResult, plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)

