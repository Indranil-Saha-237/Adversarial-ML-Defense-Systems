import numpy as np
import os
import mlflow
import joblib
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

# Config
DATA_DIR = 'data'
MODEL_PATH = 'models/baseline_rf.pkl'

def simulate_attack():
    print("Loading baseline model and data...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    model = joblib.load(MODEL_PATH)

    # Using 50 samples for depth over breadth
    X_test = np.load(f"{DATA_DIR}/X_test.npy")[:50]
    y_test = np.load(f"{DATA_DIR}/y_test.npy")[:50]

    # Step 1: Wrap for ART
    classifier = SklearnClassifier(model=model)

    # Step 2: Initialize a STRONGER HopSkipJump
    # Increasing max_iter and max_eval makes it much more likely to find a breach
    print("Step 2: Initializing Deep HopSkipJump Attack...")
    attack = HopSkipJump(
        classifier=classifier,
        max_iter=50,       # Increased from 10
        max_eval=10000,    # Increased from 100
        init_eval=100,
        verbose=True
    )

    # Step 3: Generate
    print("Step 3: Generating Adversarial Examples (Searching deep for vulnerabilities)...")
    X_test_adv = attack.generate(x=X_test)

    # Step 4: Evaluate
    clean_acc = accuracy_score(y_test, model.predict(X_test))
    adv_acc = accuracy_score(y_test, model.predict(X_test_adv))

    print("\n" + "="*40)
    print("FINAL SPRINT 2 ATTACK RESULTS")
    print(f"Clean Accuracy:       {clean_acc * 100:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc * 100:.2f}%")
    print(f"BREACH SUCCESS:       {(clean_acc - adv_acc) * 100:.2f}% Drop")
    print("="*40)

    # Log results to MLflow
    mlflow.set_experiment("Adversarial_Attacks")
    with mlflow.start_run(run_name="Strong_HopSkipJump_RF"):
        mlflow.log_param("attack_type", "HopSkipJump")
        mlflow.log_param("max_iter", 50)
        mlflow.log_metric("clean_accuracy", clean_acc)
        mlflow.log_metric("adversarial_accuracy", adv_acc)
        mlflow.log_metric("accuracy_drop", clean_acc - adv_acc)

if __name__ == "__main__":
    simulate_attack()