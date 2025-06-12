from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Import model execution and global model references
from models.knn import run_knn, knn_model, knn_scaler
from models.svm import run_svm, svm_model, svm_scaler
from models.naive_bayes import run_naive_bayes, nb_model, nb_scaler
from models.neural_network import run_neural_network, nn_model, nn_scaler
from models.linear_regression import run_linear_regression, lr_model, lr_scaler
from models.random_forest import run_random_forest, rf_model, rf_scaler

model_registry = {
    "knn": {"model": None, "scaler": None},
    "svm": {"model": None, "scaler": None},
    "naive_bayes": {"model": None, "scaler": None},
    "neural_network": {"model": None, "scaler": None},
    "linear_regression": {"model": None, "scaler": None},
    "random_forest": {"model": None, "scaler": None},
}


app = Flask(__name__)
CORS(app)

@app.route("/run_knn", methods=["POST"])
def knn():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_knn(file, target_column))

@app.route("/run_svm", methods=["POST"])
def svm():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_svm(file, target_column))

@app.route("/run_naive_bayes", methods=["POST"])
def naive_bayes():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_naive_bayes(file, target_column))

@app.route("/run_neural_network", methods=["POST"])
def neural_network():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_neural_network(file, target_column))

@app.route("/run_linear_regression", methods=["POST"])
def linear_regression():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_linear_regression(file, target_column))

@app.route("/run_random_forest", methods=["POST"])
def random_forest():
    file = request.files['file']
    target_column = request.form['target_column']
    return jsonify(run_random_forest(file, target_column))

from models.registry import model_registry

@app.route("/predict_model", methods=["POST"])
def predict_model():
    from models.registry import model_registry
    data = request.get_json()
    model_name = data.get("model")
    input_features = data.get("features")

    if not model_name or not input_features:
        return jsonify({"error": "Missing model or features"}), 400

    registry = model_registry.get(model_name)
    if not registry:
        return jsonify({"error": f"Model '{model_name}' not supported."}), 400

    model = registry.get("model")
    scaler = registry.get("scaler")
    encoders = registry.get("encoders")
    feature_names = registry.get("features")
    target_encoder = registry.get("target_encoder")

    if not model or not scaler or not feature_names:
        return jsonify({"error": "Model not found or not trained."}), 400

    if len(input_features) != len(feature_names):
        return jsonify({"error": f"Expected {len(feature_names)} features, got {len(input_features)}."}), 400

    # Encode features
    encoded_inputs = []
    for val, fname in zip(input_features, feature_names):
        try:
            # Try numeric conversion
            num_val = float(val)
            encoded_inputs.append(num_val)
        except ValueError:
            encoder = encoders.get(fname) if encoders else None
            if encoder:
                try:
                    encoded_val = encoder.transform([val])[0]
                    encoded_inputs.append(encoded_val)
                except:
                    return jsonify({"error": f"Invalid categorical value '{val}' for feature '{fname}'"}), 400
            else:
                return jsonify({"error": f"Feature '{fname}' must be numeric"}), 400

    # Predict
    X = scaler.transform([encoded_inputs])
    pred = model.predict(X)[0]

    # Decode label if categorical
    if target_encoder:
        try:
            pred = target_encoder.inverse_transform([pred])[0]
        except:
            pass  # If decoding fails, fall back to raw prediction

    return jsonify({"prediction": str(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
