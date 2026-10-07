# Network Intrusion Detection

## Project Overview

Network Intrusion Detection is a Machine Learning based cybersecurity project that detects whether network traffic is:

- **BENIGN** — Normal network traffic
- **INTRUSION** — Potential malicious or abnormal network traffic

The project uses the **CICIDS2017** dataset and a Random Forest classifier.

A Streamlit web application is also included to demonstrate predictions on network traffic samples.

---

## Objective

The main objectives of this project are:

1. Detect malicious network traffic using Machine Learning.
2. Classify network traffic into BENIGN and INTRUSION.
3. Train the model using network traffic data from multiple days.
4. Evaluate the model on a completely unseen day.
5. Provide a simple Streamlit interface for prediction.

---

## Dataset

The project uses the **CICIDS2017** network intrusion detection dataset.

The dataset contains network traffic records with multiple network-flow features.

For this project:

- **Tuesday, Wednesday and Thursday** data are used for training.
- **Friday** data is used as completely unseen test data.
- Monday was not used because it contained only BENIGN traffic.

### Dataset Labels

The original dataset contains multiple attack categories.

For this project, the problem is converted into binary classification:

| Original Label | Project Label |
|---|---:|
| BENIGN | 0 |
| Any attack/non-BENIGN traffic | 1 |

Where:

- `0 = BENIGN`
- `1 = INTRUSION`

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded network traffic data from multiple CSV files.
2. Removed unnecessary metadata columns.
3. Removed missing values.
4. Replaced infinite values.
5. Removed duplicate records.
6. Converted the original labels into binary classes.
7. Balanced the training dataset.
8. Applied feature scaling.
9. Split the data based on different days to perform stricter validation.

### Removed Metadata Columns

The following columns were removed:

```text
id
Flow ID
Src IP
Dst IP
Timestamp
Attempted Category