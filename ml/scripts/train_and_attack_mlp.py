import numpy as np
import os 
import joblib
import mlflow
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ProjectedGradientDescent

#CONFIG
DATA_DIR= 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading Processed Data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")[:100000]
    y_train = np.load(f"{DATA_DIR}/y_train.npy")[:100000]
    X_test = np.load(f"{DATA_DIR}/X_test.npy")[:5000]
    y_test = np.load(f"{DATA_DIR}/y_test.npy")[:5000]

    #1. Train MLP (The Vulnerable Target)
    print("Training MLP (Neural Network) Baseline...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=50, random_state=42)
    mlp.fit(X_train, y_train)

    clean_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"Clean Accuracy of MLP: {clean_acc * 100:.2f}%")
    joblib.dump(mlp, f"{MODEL_DIR}/baseline_mlp.pkl")

    #2. Wrap MLP with ART (requires setting clip_values because data is scaled)
    classifier = SklearnClassifier(model=mlp, clip_values=(-5.0, 5.0))

    #3. PGD Attack (The "Strong" Attack)
    print("Generating adversarial examples with PGD...")
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, max_iter=20, batch_size=32)
    X_test_adv = attack.generate(x=X_test)

    #4. Evaluate
    adv_preds = np.argmax(classifier.predict(X_test_adv), axis=1)
    adv_acc = accuracy_score(y_test, adv_preds)

    print("\n" + "="*40)
    print(f"RESULTS FOR PGD ATTACK")
    print(f"Clean Accuracy:       {clean_acc * 100:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc * 100:.2f}%")
    print(f"Accuracy Drop:        {(clean_acc - adv_acc) * 100:.2f}%")
    print("="*40)

    #LOG RESULTS TO MLflow
    mlflow.set_experiment("Sprint_2_Core_Attacks")
    with mlflow.start_run(run_name='MLP_PGD_Attack'):
        mlflow.log_metric("clean_accuracy", clean_acc)
        mlflow.log_metric("adversarial_accuracy", adv_acc)
        mlflow.log_param("attack_type", "PGD")

if __name__ == "__main__":
    main()

