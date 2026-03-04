import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import HopSkipJump

# Config
DATA_DIR = 'data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model save path outside the MLflow block to avoid NameError on failure
MODEL_SAVE_PATH = f"{MODEL_DIR}/defended_rf.pkl"


def train_defended():
    print("--- [ ADVERSARIAL TRAINING INITIALIZED ] ---")

    # 1. Load Data
    print("Loading preprocessed data...")
    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    X_test = np.load(f"{DATA_DIR}/X_test.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    # 2. Load Baseline Model
    baseline_path = f"{MODEL_DIR}/baseline_rf.pkl"
    if not os.path.exists(baseline_path):
        print(f"Error: {baseline_path} not found. Please train the baseline first.")
        return
    baseline_model = joblib.load(baseline_path)

    # 3. Generate Adversarial Examples for Training
    # Reduced subset size: HopSkipJump is query-intensive (~max_eval queries per sample).
    # 20,000 samples x 1,000 evals = ~20M model queries, which is infeasible.
    # Use 1,000 samples for training adversarial augmentation.
    SUBSET_SIZE = 1000
    X_train_sub = X_train[:SUBSET_SIZE]
    y_train_sub = y_train[:SUBSET_SIZE]

    # Compute global clip values to constrain adversarial perturbations
    # to the valid feature space and prevent out-of-distribution examples.
    clip_min = float(X_train.min())
    clip_max = float(X_train.max())

    print(f"Step 1: Wrapping baseline with ART (clip_values=[{clip_min}, {clip_max}]) "
          f"and generating {SUBSET_SIZE} adversarial samples...")

    # Fix 1: Pass clip_values so ART constrains perturbations to valid feature range.
    baseline_classifier = SklearnClassifier(
        model=baseline_model,
        clip_values=(clip_min, clip_max)
    )

    # Initialize HopSkipJump against the baseline to craft training-time adversarial examples.
    attack_train = HopSkipJump(
        classifier=baseline_classifier,
        max_iter=10,
        max_eval=1000,
        init_eval=100,
        verbose=False
    )
    X_train_adv = attack_train.generate(x=X_train_sub)

    # 4. Data Augmentation
    print("Step 2: Augmenting dataset (Clean Data + Adversarial Trap Examples)...")
    # Stack full clean training set with the adversarial subset.
    # y_train_sub is the correct label array for X_train_adv (same indices, same size).
    X_augmented = np.vstack([X_train, X_train_adv])
    y_augmented = np.concatenate([y_train, y_train_sub])

    # 5. Training the Defended Model
    print("Step 3: Training Robust Random Forest (Adversarially Trained)...")
    mlflow.set_experiment("Defense_Benchmarks")

    with mlflow.start_run(run_name="RF_Adversarial_Training") as run:
        try:
            defended_rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                n_jobs=-1,
                random_state=42
            )
            defended_rf.fit(X_augmented, y_augmented)

            # 6. Benchmarking Robustness
            print("Step 4: Benchmarking defense robustness...")

            # Clean Accuracy on full test set
            clean_acc = accuracy_score(y_test, defended_rf.predict(X_test))

            # Fix 2: Wrap the DEFENDED model (not baseline) for adversarial evaluation,
            # so the attack is crafted against the model actually being assessed.
            TEST_SUBSET_SIZE = 500  # Keep evaluation feasible
            X_test_sub = X_test[:TEST_SUBSET_SIZE]
            y_test_sub = y_test[:TEST_SUBSET_SIZE]

            defended_classifier = SklearnClassifier(
                model=defended_rf,
                clip_values=(clip_min, clip_max)
            )
            attack_eval = HopSkipJump(
                classifier=defended_classifier,
                max_iter=10,
                max_eval=1000,
                init_eval=100,
                verbose=False
            )
            X_test_adv = attack_eval.generate(x=X_test_sub)
            adv_acc = accuracy_score(y_test_sub, defended_rf.predict(X_test_adv))

            print("\n" + "=" * 40)
            print("DEFENSE PERFORMANCE")
            print(f"Clean Accuracy:        {clean_acc * 100:.2f}%")
            print(f"Adversarial Accuracy:  {adv_acc * 100:.2f}%")
            print("=" * 40)

            # Fix 3: Log accurate params reflecting the actual attack used (HopSkipJump).
            mlflow.log_param("defense_method", "Adversarial_Training_HopSkipJump")
            mlflow.log_param("train_subset_size", SUBSET_SIZE)
            mlflow.log_param("test_subset_size", TEST_SUBSET_SIZE)
            mlflow.log_param("max_iter", 10)
            mlflow.log_param("max_eval", 1000)
            mlflow.log_param("clip_min", clip_min)
            mlflow.log_param("clip_max", clip_max)
            mlflow.log_metric("clean_accuracy", clean_acc)
            mlflow.log_metric("adversarial_accuracy", adv_acc)

            # Save the hardened model
            joblib.dump(defended_rf, MODEL_SAVE_PATH)
            mlflow.log_artifact(MODEL_SAVE_PATH)
            mlflow.sklearn.log_model(defended_rf, "model")

            print(f"\nSuccess! Defended model saved to {MODEL_SAVE_PATH}")

        except Exception as e:
            # Fix 4: Tag the run as failed so it doesn't linger as "RUNNING" in MLflow UI.
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error", str(e))
            print(f"\nError during training: {e}")
            raise


if __name__ == "__main__":
    train_defended()