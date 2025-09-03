from flask import Flask
from flask import render_template  
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/FiltradoDeCorreos')
def Filtrado_de_correos():
    return render_template('FiltradoDeCorreos.html')

@app.route('/FraudesBancarios')
def FraudesBancarios():
    return render_template('FraudesBancarios.html')

@app.route('/PagLugo')
def pag_lugo():
    return render_template('PagLugo.html')


# Ruta para la página de Camero
@app.route('/PagCamero')
def pag_camero():
    return render_template('PagCamero.html')

if __name__ == '__main__':
    app.run(debug=True)





