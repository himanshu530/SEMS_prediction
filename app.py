from flask import Flask,request,render_template
import pandas as pd 
import numpy as np 
import pickle
import joblib
import flasgger
from flasgger import Swagger  
import datetime

app = Flask(__name__)

file = open('linear_model.pkl','rb')
ml_model = joblib.load(file)

@app.route("/")

def home():
    print("This is the homepage")

    return render_template('index.html')

@app.route("/predict", methods=["POST"])

def predict():
    if request.method == "POST":
        print(request.form)
        
    dt = datetime.datetime.strptime(
                     request.form['startdate'],
                     '%Y-%m-%d')
    print(dt)
    dt = dt.toordinal()
    dt = np.array(dt).reshape(-1,1)

    prediction = ml_model.predict(dt)
    print(prediction)
    prediction = str(prediction).lstrip('[]').rstrip(']')
    return render_template("output.html", prediction = prediction)
    


if __name__ == '__main__':
    app.run(debug=True)