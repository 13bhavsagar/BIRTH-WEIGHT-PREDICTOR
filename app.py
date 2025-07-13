
from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle as pk
app=Flask(__name__)

def cleaned_data(form_data):
    gestation=float(form_data['gestation'])
    parity=float(form_data['parity'])
    age=float(form_data['age'])
    height=float(form_data['height'])
    weight=float(form_data['weight'])
    smoke=float(form_data['smoke'])
    cleaned_data={"gestation":[gestation],
                  "parity":[parity],
                  "age":[age],
                  "height":[height],
                  "weight":[weight],
                  "smoke":[smoke]
                  }
    return cleaned_data

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def get_prediction():
    baby_data=request.form
    data=cleaned_data(baby_data)
    df=pd.DataFrame(data)
    with open('model/model.pkl', 'rb') as obj:
        model = pk.load(obj)

    prediction=model.predict(df) 
    prediction=float(prediction[0])
    prediction = round(prediction, 2)
    response={
        "prediction": prediction
    }  
    return render_template('index.html',prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
