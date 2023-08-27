from flask import Flask,request,jsonify
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import json

scaler = StandardScaler()
scaler.mean_ =  np.array([427.68,2079.48133333,1759.656])
scaler.scale_ = np.array([482.54803036,613.41137392,648.2068407])  

#Loading ML model using pickle
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

#Routing : HOME
@app.route('/')
def index():
    return "GRAIN ANALYSIS API"

#Routing:Prediction
@app.route('/predict',methods=['GET','POST'])
def predict():
    #Recieving JSON values
    g1=(float)(request.json['Gas1'])
    g2=(float)(request.json['Gas2'])
    g3=(float)(request.json['Gas3'])

    #Transforming the input values
    arr = [[g1,g2,g3]]
    input_query = np.array(arr)
    input_process=scaler.transform(input_query)

    #Predict
    result = model.predict(input_process)[0]
    prob=model.predict_proba(input_process)

    #return prediction results in json format
    return jsonify({'FIT':'YES' if(result) else 'NO',
                    'P[Not_FIT]':str(np.round(prob[0][0],2)),
                    'P[FIT]': str(np.round(prob[0][1],2))})

if __name__ == '__main__':
    app.run(debug=True)