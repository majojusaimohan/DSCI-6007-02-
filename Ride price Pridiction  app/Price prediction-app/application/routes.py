from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import requests
import numpy
import pandas as pd

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

cabprovider = 0
CabName = 9
Distance=0
Surge=0
Visibility=0

#decorator to access the service
@app.route("/RidePricePrediction", methods=['GET', 'POST'])
def RidePricePrediction():
    
    #extract form inputs
    cabprovider = request.form.get("cabprovider")
    print(cabprovider)
    CabName = request.form.get("CabName")
    Distance = request.form.get("Distance")
    Surge = request.form.get("Surge")
    Visibility = request.form.get("Visibility")
    listofzeros = [0] * 18
    listofzeros[0]=Distance
    listofzeros[1]=Surge
    listofzeros[2]=Visibility
    listofzeros[int(0 if cabprovider is None else cabprovider)]=1
    listofzeros[int(0 if CabName is None else CabName)]=1
    print(listofzeros)
    url = "http://localhost:3000/api" 
    #post data to url
    input_data = json.dumps(listofzeros)
    results = requests.post(url, input_data)
    print(input_data)
    #send input values and prediction result to index.html for display
    return render_template("index.html", cabprovider = cabprovider, CabName = CabName, Distance = Distance, Surge = Surge, Visibility = Visibility,   results=results.content.decode('UTF-8'))


  
