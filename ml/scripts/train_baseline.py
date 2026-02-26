import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#CONFIG
DATA_DIR= 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print("Loading preprocessed data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    print(f"Training on {X_train.shape[0]:,} samples with {X_train.shape[1]} features...")

    #Start ML flow
    mlflow.set_experiment("IDS2018_Baselines")
    with mlflow.start_run(run_name='RandomForest_Baseline'):

    # Initialize Model (n_jobs=-1 uses all CPU cores for speed) 
    # Using a small n_estimators for the very first run to ensure   
        rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
        print("Fitting Model (May take a few minutes)...")
        rf.fit(X_train, y_train) 

        print("Evaluating Model...")
        y_pred = rf.predict(X_test)

        #Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"Baseline Accuracy: {acc:.4f}")

        #Log to ML flow
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("attack_precision", report['1']['f1-score'])

        #Save Model
        model_path = f"{MODEL_DIR}/baseline_rf.pkl"
        joblib.dump(rf, model_path)
        mlflow.log_artifact(model_path)

        #Log the sklearn model directly to MLflow
        mlflow.sklearn.log_model(rf, "model")

    print(f"Training complete. Model saved to {model_path} and logged to MLflow.")
if __name__ == "__main__":
    train() 
    
