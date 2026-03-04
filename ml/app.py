from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import random
import os
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

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

@app.route('/attack/simulate', methods=['POST'])
def simulate_live_attack():
    #1. Load a small slice of data (100 samples)
    try: 
        X_full = np.load("data/X_test.npy")
        y_full = np.load("data/y_test.npy")

        random_indices= random.sample(range(len(X_full)), 10)

        X_test = X_full[random_indices]
        y_test = y_full[random_indices]



        #2. Load the models
        baseline_model = joblib.load(BASELINE_PATH)
        defended_model = joblib.load(DEFENDED_PATH)

        #3. Wrap models with ART Attack
        base_clf = SklearnClassifier(model=baseline_model)
        attack = HopSkipJump(classifier=base_clf, max_iter=5, max_eval=100, init_eval=10, verbose=False)
        X_adv = attack.generate(x=X_test)

        #4. Evaluate both models on the SAME attack
        from sklearn.metrics import accuracy_score
        baseline_adv_acc = accuracy_score(y_test, baseline_model.predict(X_adv))
        defended_adv_acc = accuracy_score(y_test, defended_model.predict(X_adv))

        return jsonify({
            "status": "success",
            "samples_tested": 10,
            "baseline_accuracy" : float(baseline_adv_acc),
            "defended_accuracy" : float(defended_adv_acc)
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)

