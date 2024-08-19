from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

# Load the trained model
model = mlflow.sklearn.load_model("svm_model")

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello there!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    predictions = model.predict(data['input'])
    print(predictions)
    return jsonify(predictions=predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
