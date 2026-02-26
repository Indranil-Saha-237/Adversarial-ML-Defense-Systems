from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app) #Allows Node.js(PORT 3000) to access the API

#config
MODEL_PATH = "models/baseline_rf.pkl"

@app.route('/health', methods = ['GET'])
def health():
    return jsonify({"status": "online", "engine": "RandomForest", "accuracy": 0.998})
@app.route('/model/metadata', methods=['GET'])
def get_metadata():
    if os.path.exists(MODEL_PATH):

        return jsonify({
                "model_name": "Random Forest Baseline",
                "accuracy": 0.9999,
                "precision": 0.9991,
                "features_count": 69,
                "status": "Ready for Deployment"
        })
    else:
        return jsonify({"error": "Model not found"}), 404
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

