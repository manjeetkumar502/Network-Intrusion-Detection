import streamlit as st
import numpy as np
import joblib
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "strict_random_forest.joblib"
)


SAMPLE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "strict_test_samples.npy"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model


model = load_model()


# ==========================================
# LOAD TEST SAMPLES
# ==========================================

@st.cache_data
def load_samples():

    samples = np.load(
        SAMPLE_PATH
    )

    return samples


# ==========================================
# TITLE
# ==========================================

st.title(
    "🛡️ Network Intrusion Detection System"
)

st.write(
    "A Machine Learning based system for detecting "
    "whether network traffic is **BENIGN** or an "
    "**INTRUSION**."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header(
    "🔧 Model Information"
)

st.sidebar.success(
    "Random Forest"
)

st.sidebar.write(
    "Classification: Binary"
)

st.sidebar.write(
    "0 → BENIGN"
)

st.sidebar.write(
    "1 → INTRUSION"
)

st.sidebar.divider()

st.sidebar.write(
    "**Validation Method**"
)

st.sidebar.write(
    "Training: Tuesday + Wednesday + Thursday"
)

st.sidebar.write(
    "Testing: Friday"
)

st.sidebar.write(
    "Friday data was completely unseen during training."
)


# ==========================================
# NETWORK TRAFFIC PREDICTION
# ==========================================

st.header(
    "🔍 Network Traffic Prediction"
)

st.write(
    "Test a randomly selected network traffic sample "
    "from the unseen Friday dataset."
)


if st.button(
    "🚀 Test Random Network Sample"
):

    try:

        samples = load_samples()

        # Select random sample
        random_index = np.random.randint(
            0,
            len(samples)
        )

        sample = samples[
            random_index
        ].reshape(
            1,
            -1
        )


        # ==================================
        # PREDICTION
        # ==================================

        prediction = model.predict(
            sample
        )[0]


        probability = model.predict_proba(
            sample
        )[0]


        benign_probability = (
            probability[0] * 100
        )


        intrusion_probability = (
            probability[1] * 100
        )


        # ==================================
        # RESULT
        # ==================================

        st.subheader(
            "Prediction Result"
        )


        if prediction == 1:

            st.error(
                "🚨 INTRUSION DETECTED"
            )

            st.metric(
                "Intrusion Probability",
                f"{intrusion_probability:.2f}%"
            )

        else:

            st.success(
                "✅ BENIGN TRAFFIC"
            )

            st.metric(
                "Benign Probability",
                f"{benign_probability:.2f}%"
            )


        st.caption(
            f"Test sample index: {random_index}"
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

st.divider()

st.header(
    "📊 Model Performance"
)


col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "Accuracy",
        "99.61%"
    )


with col2:

    st.metric(
        "Precision",
        "99.99%"
    )


with col3:

    st.metric(
        "Recall",
        "99.18%"
    )


with col4:

    st.metric(
        "F1 Score",
        "99.58%"
    )


# ==========================================
# VALIDATION METHOD
# ==========================================

st.divider()

st.header(
    "🧪 Validation Method"
)

st.write(
    """
    The model was evaluated using strict day-based validation.
    Network traffic from Tuesday, Wednesday and Thursday was
    used for training, while Friday traffic was kept completely
    separate for testing.
    """
)


col1, col2 = st.columns(2)


with col1:

    st.info(
        "📚 Training Data\n\n"
        "Tuesday\n\n"
        "Wednesday\n\n"
        "Thursday"
    )


with col2:

    st.warning(
        "🧪 Unseen Test Data\n\n"
        "Friday\n\n"
        "Not used during training"
    )


# ==========================================
# CONFUSION MATRIX RESULTS
# ==========================================

st.header(
    "📌 Confusion Matrix Results"
)


col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "True Negatives",
        "288,506"
    )


with col2:

    st.metric(
        "False Positives",
        "38"
    )


with col3:

    st.metric(
        "False Negatives",
        "2,121"
    )


with col4:

    st.metric(
        "True Positives",
        "256,892"
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

st.divider()

st.header(
    "📌 About the Project"
)

st.write(
    """
    The Network Intrusion Detection System is a Machine Learning
    based cybersecurity project designed to identify malicious
    network traffic.

    The system uses network traffic features and a Random Forest
    classifier to classify traffic into two categories:

    • BENIGN – Normal network activity

    • INTRUSION – Potentially malicious network activity
    """
)


st.info(
    "Dataset: CICIDS2017 | Model: Random Forest"
)


# ==========================================
# LIMITATION
# ==========================================

st.header(
    "⚠️ Current Limitation"
)

st.write(
    """
    The current application demonstrates intrusion detection using
    prepared network traffic samples. It does not directly capture
    live packets from a network interface.
    """
)