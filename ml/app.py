from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app) #Allows Node.js(PORT 3000) to access the API
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

#config
BASELINE_PATH = "models/baseline_rf.pkl"
DEFENDED_PATH = "models/defended_rf.pkl"

@app.route('/health', methods=['GET'])
def health():
    # reports if defense layer is physically present
    defense_ready = os.path.exists(DEFENDED_PATH)
    return jsonify({
        "status": "online",
        "engine": "RandomForest",
        "defense_active": defense_ready
    })

@app.route('/model/metadata', methods = ['GET'])
def get_metadata():
    if os.path.exists(DEFENDED_PATH):
        return jsonify({
                "model_name": "hardened Random Forest",
                "accuracy": 0.9999,
                "adversarial_accuracy": 0.9260,
                "features_count": 69,
                "status": "SECURED",
                "defense_type": "Adversarial Training"
        })
    elif os.path.exists(BASELINE_PATH):
        return jsonify({
                "model_name": "Baseline Random Forest",
                "accuracy": 0.9999,
                "adversarial_accuracy": 0.10,
                "features_count": 69,
                "status": "VULNERABLE",
                "defense_type": "None"
        })
    else:
        return jsonify({"error": "No model files found in /models"}), 404

@app.route('/model/comparison', methods=['GET'])
def get_comparison():
    return jsonify({
        "labels": ["Clean Accuracy", "Adversarial Accuracy"],
        "baseline": [0.9999, 0.10],
        "defended": [0.9999, 0.9260]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

