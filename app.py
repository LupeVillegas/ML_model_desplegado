from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/", methods = ["GET"]) # landing page (endpoint /)
def hello():
    return """
    <html>
        <head>
            <title>Classification API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                table { border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ccc; padding: 8px 12px; }
                th { background-color: #f4f4f4; }
            </style>
        </head>
        <body>
            <h1>Welcome to my Classification API</h1>
            <h2>Info</h2>
            <p>Input Features: <b>area, length, diameter, external_area, area_ratio</b></p>
            <h3>Classes</h3>
            <table>
                <tr><th>Label</th><th>Name</th></tr>
                <tr><td>0</td><td>cell type 0</td></tr>
                <tr><td>1</td><td>cell type 1</td></tr>
                <tr><td>2</td><td>cell type 2</td></tr>
                <tr><td>3</td><td>tumoral cells</td></tr>
            </table>
            <h3>Usage Example</h3>
            <p><b>URL:</b> <code>/api/predict?area=78.0&length=40.0&diameter=12.0&external_area=94.0&area_ratio=0.85</code></p>
        </body>
    </html>
    """


@app.route("/api/predict", methods = ["GET"])# endpoint '/api/predict', método GET
def predict(): 
    with open('model_morphology.pkl', 'rb') as f:
        model = pickle.load(f)

    fields = ["area", "length", "diameter", "external_area", "area_ratio"]
    try:
        features = []
        for field in fields:
            value = request.args.get(field, None)
            if value is None:
                return jsonify({"error": "Args empty, not enough data to predict}"}), 400
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({"error": f"Feature {field} must be numeric"}), 400

    
        # Convert into dataframe
        features_df = pd.DataFrame([features], columns=fields)
        print(f"Input features:{features}")

        #Prediction
        prediction = int(model.predict(features_df)[0])   
        probabilities = model.predict_proba(features_df)[0]*100 
        probability=np.round(probabilities[prediction:prediction+1][0],2)

        print(f"Prediction -> cell type:{prediction}")
        print(f"Preobability -> {probability}%")
        
        return jsonify({
            "prediction(cellType)": prediction,        
            "probability(%)": probability
        })

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)