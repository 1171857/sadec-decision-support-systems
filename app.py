import numpy as np
from flask import Flask, request, render_template 
import os
from breastCancer import BreastCancer 

app = Flask(__name__, template_folder='templates')
path = os.path.dirname(os.path.abspath(__file__))
br = BreastCancer(path, 3, 0.3, 15)
globals = {'node': 0, 'response': '', 'is_leaf': False}
   
@app.route('/')
def home():
    globals["node"] = 0
    globals["response"] = ''
    globals["is_leaf"] = False
    return render_template('index.html', question=br.get_Question(0))

@app.route('/predict', methods=['POST'])
def predict():

    return render_template('index.html', prediction_text=f'Diagnosis result {result}')

@app.route('/questions', methods=['POST'])
def questions():
    globals["response"] = request.form.get('response')
    try:
        exec(br.get_logic(),globals)
    except SystemExit:
        pass
    if globals["is_leaf"]:
        return render_template('index.html', result=br.get_Result(globals["node"]))
    return render_template('index.html', question=br.get_Question(globals["node"]))

if __name__ == "__main__":
    app.run()