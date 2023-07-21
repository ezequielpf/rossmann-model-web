import pandas as pd
import pickle
from flask import Flask, request, Response, jsonify
import json
import os

from rossmann.Rossmann import Rossmann

# loading model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))

# initialize API
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
     
    test_json = request.get_json()
   
    test_raw = pd.json_normalize(json.loads(test_json))

    pipeline = Rossmann()

    df1 = pipeline.data_cleaning(test_raw)
    df2 = pipeline.feature_engineering(df1)
    df3 = pipeline.data_preparation(df2)
    df_response = pipeline.get_prediction(model, test_raw, df3)
    return df_response

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(port=port, host='0.0.0.0', debug=False)
