# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

# Define title of app
@app.route('/', methods=['GET'])
def home():
    return "<h1>Predict housing prices</h1><p>This site is a prototype API for predicting housing prices.</p>"


# predict price using the linear model  
@app.route('/predict', methods=['POST'])
def predict():
    if lm:
        try: 
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)

            #prediction = lm.predict(query)
            std = std
            print(std)
			
            #return jsonify({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction), 'std': str(std)})
            #return jsonify({'std': str(std)})	

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
		
# load the linear model and the std 
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lm = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    std = joblib.load("std.pkl") # Load "model.pkl"
    print ('std loaded')

app.run(port=port, debug=True)
