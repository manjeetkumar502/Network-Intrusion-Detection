import pandas as pd
import numpy as np
import os

# ==========================================
# 1. Dataset location
# ==========================================

DATA_PATH = r"D:\network intrussion"

OUTPUT_PATH = r"data\processed\intrusion_dataset.csv"


# ==========================================
# 2. Input files
# ==========================================

files = [
    "tuesday.csv",
    "wednesday.csv",
    "thursday.csv",
    "friday.csv"
]


# ==========================================
# 3. Columns to remove
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
# 4. Start
# ==========================================

print("Starting dataset preparation...\n")

dataframes = []


# ==========================================
# 5. Read datasets
# ==========================================

for file in files:

    file_path = os.path.join(DATA_PATH, file)

    print(f"Reading: {file}")

    df = pd.read_csv(file_path)

    print(f"Rows loaded: {len(df):,}")

    # BENIGN = 0
    # Any attack = 1

    df["Label"] = df["Label"].apply(
        lambda x: 0
        if str(x).strip().upper() == "BENIGN"
        else 1
    )

    dataframes.append(df)


# ==========================================
# 6. Combine datasets
# ==========================================

df = pd.concat(
    dataframes,
    ignore_index=True
)

print(
    f"\nTotal rows before cleaning: {len(df):,}"
)


# ==========================================
# 7. Remove unnecessary columns
# ==========================================

df = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)


# ==========================================
# 8. Replace infinite values
# ==========================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ==========================================
# 9. Remove missing values
# ==========================================

df = df.dropna()

print(
    f"Rows after removing missing values: {len(df):,}"
)


# ==========================================
# 10. Remove duplicate rows
# ==========================================

before_duplicates = len(df)

df = df.drop_duplicates()

duplicates_removed = before_duplicates - len(df)

print(
    f"Duplicate rows removed: {duplicates_removed:,}"
)

print(
    f"Rows after removing duplicates: {len(df):,}"
)


# ==========================================
# 11. Separate classes
# ==========================================

benign = df[df["Label"] == 0]

intrusion = df[df["Label"] == 1]

print(
    f"\nBENIGN rows: {len(benign):,}"
)

print(
    f"INTRUSION rows: {len(intrusion):,}"
)


# ==========================================
# 12. Balance the dataset
# ==========================================

sample_size = min(
    len(benign),
    len(intrusion),
    100000
)

print(
    f"\nSamples selected per class: {sample_size:,}"
)


benign_sample = benign.sample(
    n=sample_size,
    random_state=42
)

intrusion_sample = intrusion.sample(
    n=sample_size,
    random_state=42
)


# ==========================================
# 13. Combine balanced data
# ==========================================

final_df = pd.concat(
    [
        benign_sample,
        intrusion_sample
    ],
    ignore_index=True
)


# ==========================================
# 14. Shuffle
# ==========================================

final_df = final_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================
# 15. Create output directory
# ==========================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ==========================================
# 16. Save processed dataset
# ==========================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ==========================================
# 17. Final results
# ==========================================

print("\n===================================")
print("DATASET PREPARATION COMPLETED")
print("===================================")

print(
    f"\nFinal dataset shape: {final_df.shape}"
)

print("\nFinal class distribution:")

print(
    final_df["Label"].value_counts()
)

print("\nLabel meaning:")

print("0 = BENIGN")
print("1 = INTRUSION")

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)