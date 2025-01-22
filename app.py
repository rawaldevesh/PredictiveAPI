from flask import Flask, request, jsonify
import pandas as pd
from model import train_model, make_prediction

app = Flask(__name__)

# Route to upload dataset
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save('dataset/uploaded_data.csv')
    return jsonify({"message": "File uploaded successfully!"})

# Route to train the model
@app.route('/train', methods=['POST'])
def train():
    data = pd.read_csv('dataset/uploaded_data.csv')
    metrics = train_model(data)
    return jsonify({"message": "Model trained successfully!", "metrics": metrics})

# Route to predict
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    prediction = make_prediction(input_data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)
