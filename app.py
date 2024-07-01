from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

def predict(values, model_path):
    model = load_model(model_path)
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, 'models/diabetes.h5')
    except Exception as e:
        message = f"Error: {str(e)}"
        return render_template("home.html", message=message)

    return render_template('predict.html', pred=pred)

@app.route("/malariapredict", methods=['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)
                pred = predict(img, "models/malaria.h5")
        except Exception as e:
            message = f"Error: {str(e)}"
            return render_template('malaria.html', message=message)
    return render_template('malaria_predict.html', pred=pred)

@app.route("/pneumoniapredict", methods=['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0
                pred = predict(img, "models/pneumonia.h5")
        except Exception as e:
            message = f"Error: {str(e)}"
            return render_template('pneumonia.html', message=message)
    return render_template('pneumonia_predict.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)
