import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Load prepared data
# ==========================================

MODEL_DIR = r"models"

print("Loading training and testing data...\n")

X_train = np.load(
    os.path.join(MODEL_DIR, "X_train.npy")
)

X_test = np.load(
    os.path.join(MODEL_DIR, "X_test.npy")
)

y_train = np.load(
    os.path.join(MODEL_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(MODEL_DIR, "y_test.npy")
)


print(f"X_train: {X_train.shape}")
print(f"X_test:  {X_test.shape}")


# ==========================================
# 2. Define models
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
}


# ==========================================
# 3. Train and evaluate
# ==========================================

results = {}

print("\n===================================")
print("MODEL TRAINING STARTED")
print("===================================\n")


for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test
    )

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    print(f"\n{name} Results")
    print("-----------------------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "BENIGN",
                "INTRUSION"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


# ==========================================
# 4. Select best model
# ==========================================

best_name = max(
    results,
    key=lambda name: results[name]["f1"]
)

best_model = results[best_name]["model"]

best_f1 = results[best_name]["f1"]


print("\n===================================")
print("BEST MODEL")
print("===================================")

print(
    f"Model: {best_name}"
)

print(
    f"F1 Score: {best_f1:.4f}"
)


# ==========================================
# 5. Save best model
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "best_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)


print(
    f"\nBest model saved to:"
)

print(
    model_path
)


# ==========================================
# 6. Save model name
# ==========================================

with open(
    os.path.join(
        MODEL_DIR,
        "model_name.txt"
    ),
    "w"
) as f:

    f.write(best_name)


print("\nModel training completed successfully!")