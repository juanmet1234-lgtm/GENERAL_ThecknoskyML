from flask import Flask
from flask import render_template  
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 



if __name__ == '__main__':
    app.run(debug=True)

@app.route('/PagLugo')
def pag_lugo():
    return render_template('PagLugo.html')





