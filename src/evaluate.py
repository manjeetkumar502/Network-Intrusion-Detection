import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = r"D:\network intrussion"

TRAIN_FILES = [
    "tuesday.csv",
    "wednesday.csv",
    "thursday.csv"
]

TEST_FILE = "friday.csv"

MODEL_PATH = r"models\strict_random_forest.joblib"
SCALER_PATH = r"models\strict_scaler.joblib"
SAMPLE_PATH = r"models\strict_test_samples.npy"


# ==========================================
# COLUMNS TO REMOVE
# ==========================================

DROP_COLUMNS = [
    "id",
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Timestamp",
    "Attempted Category"
]


# ==========================================
# LOAD DATA
# ==========================================

def load_data(file_list):

    dataframes = []

    for file in file_list:

        path = os.path.join(DATA_PATH, file)

        print(f"Reading: {file}")

        df = pd.read_csv(
            path,
            low_memory=False
        )

        print(f"Rows loaded: {len(df):,}")

        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True
    )


# ==========================================
# PREPARE DATA
# ==========================================

def prepare_data(df):

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    feature_columns = [
        col for col in df.columns
        if col != "Label"
    ]

    for col in feature_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    df["Label"] = df["Label"].apply(
        lambda x:
        0 if str(x).strip().upper() == "BENIGN"
        else 1
    )

    X = df.drop(
        columns=["Label"]
    )

    y = df["Label"]

    return X, y


# ==========================================
# START
# ==========================================

print("\n===================================")
print("STRICT DAY-BASED VALIDATION")
print("===================================\n")


# ==========================================
# TRAINING DATA
# ==========================================

print("Loading training data...")

train_df = load_data(
    TRAIN_FILES
)

print(
    f"\nTotal training rows: "
    f"{len(train_df):,}"
)

X_train, y_train = prepare_data(
    train_df
)

print(
    f"Training features: "
    f"{X_train.shape}"
)

print(
    f"Training labels: "
    f"{y_train.shape}"
)

print("\nTraining class distribution:")

print(
    y_train.value_counts()
)


# ==========================================
# BALANCE TRAINING DATA
# ==========================================

print("\nBalancing training data...")

train_data = X_train.copy()

train_data["Label"] = y_train.values

benign = train_data[
    train_data["Label"] == 0
]

intrusion = train_data[
    train_data["Label"] == 1
]

sample_size = min(
    len(benign),
    len(intrusion),
    100000
)

benign = benign.sample(
    n=sample_size,
    random_state=42
)

intrusion = intrusion.sample(
    n=sample_size,
    random_state=42
)

train_data = pd.concat(
    [benign, intrusion],
    ignore_index=True
)

train_data = train_data.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)

y_train = train_data["Label"]

X_train = train_data.drop(
    columns=["Label"]
)

print(
    f"Balanced training samples: "
    f"{len(X_train):,}"
)


# ==========================================
# LOAD FRIDAY TEST DATA
# ==========================================

print("\n===================================")
print("LOADING UNSEEN FRIDAY DATA")
print("===================================\n")

test_path = os.path.join(
    DATA_PATH,
    TEST_FILE
)

print(
    f"Reading: {TEST_FILE}"
)

test_df = pd.read_csv(
    test_path,
    low_memory=False
)

print(
    f"Rows loaded: "
    f"{len(test_df):,}"
)

X_test, y_test = prepare_data(
    test_df
)

print(
    f"Test features: "
    f"{X_test.shape}"
)

print(
    f"Test labels: "
    f"{y_test.shape}"
)

print("\nFriday class distribution:")

print(
    y_test.value_counts()
)


# ==========================================
# MATCH FEATURES
# ==========================================

X_test = X_test[
    X_train.columns
]


# ==========================================
# SCALING
# ==========================================

print("\nScaling data...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# SAVE SAMPLE DATA FOR APP
# ==========================================

sample_count = min(
    1000,
    len(X_test_scaled)
)

sample_indices = np.random.RandomState(
    42
).choice(
    len(X_test_scaled),
    size=sample_count,
    replace=False
)

X_test_samples = X_test_scaled[
    sample_indices
]

np.save(
    SAMPLE_PATH,
    X_test_samples
)

print(
    f"\nSaved {sample_count} correctly scaled "
    f"Friday samples for the Streamlit app."
)

print(
    f"Sample file: {SAMPLE_PATH}"
)


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\n===================================")
print("TRAINING RANDOM FOREST")
print("===================================\n")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

print(
    "Training completed."
)


# ==========================================
# PREDICTIONS
# ==========================================

print(
    "\nGenerating predictions on "
    "unseen Friday data..."
)

y_pred = model.predict(
    X_test_scaled
)


# ==========================================
# PERFORMANCE
# ==========================================

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


print("\n===================================")
print("STRICT VALIDATION PERFORMANCE")
print("===================================\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================\n")

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


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# SAVE MODEL AND SCALER
# ==========================================

print("\n===================================")
print("SAVING STRICT MODEL")
print("===================================\n")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    f"Model saved to:\n{MODEL_PATH}"
)

print(
    f"Scaler saved to:\n{SCALER_PATH}"
)


# ==========================================
# FINAL
# ==========================================

print("\n===================================")
print("STRICT VALIDATION COMPLETED")
print("===================================")

print(
    "\nTraining data:"
)

print(
    "Tuesday + Wednesday + Thursday"
)

print(
    "\nTesting data:"
)

print(
    "Friday"
)

print(
    "\nFriday data was NOT used during training."
)

print(
    "\nStrict model, scaler, and correctly "
    "scaled test samples are ready for the app."
)