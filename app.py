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
    list_values = []
    list_values.append(float(request.form.get('Radius')))
    list_values.append(float(request.form.get('Texture')))
    list_values.append(float(request.form.get('Perimeter')))
    list_values.append(float(request.form.get('Area')))
    list_values.append(float(request.form.get('Smoothness')))
    list_values.append(float(request.form.get('Compactness')))
    list_values.append(float(request.form.get('Concavity')))
    list_values.append(float(request.form.get('Concave Points')))
    list_values.append(float(request.form.get('Symmetry')))
    list_values.append(float(request.form.get('Fractal Dimension')))

    result = br.get_Results_From_All_Models(list_values)

    return render_template('index.html', Diagnosis_result=f'Diagnosis result.',lrm_result=f'lrm: {result[0]}' , 
                    dtm_result=f'dtm: {result[1]}', nbm_result=f'nbm: {result[2]}' ,knn_result=f'knn: {result[3]}',
                    svm_result=f'svm: {result[4]}', question=br.get_Question(globals["node"]) )

@app.route('/questions', methods=['POST'])
def questions():
    globals["response"] = request.form.get('response')
    try:
        exec(br.get_logic(),globals)
    except SystemExit:
        pass
    if globals["is_leaf"]:
        return render_template('index.html', result_title=f'Results.', result=br.get_Result(globals["node"]))
    return render_template('index.html', question=br.get_Question(globals["node"]))

if __name__ == "__main__":
    app.run()