
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FedCare AI | Federated Learning",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "federated_model.pkl"
RESULTS_PATH = BASE_DIR / "results.json"
TEST_PATH = BASE_DIR / "global_test.csv"

TARGET = "Diagnosis"

# Confirm that these labels match the encoding used during training.
LABELS = {
    0: "Benign",
    1: "Malignant",
}

DEVELOPER_NAME = "Rahma Ashraf Ali Faragallah"
DEVELOPER_ROLE = (
    "AI Lead | Machine Learning | Deep Learning | Federated Learning"
)


# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(
            135deg, #07111F 0%, #0B1728 55%, #102438 100%
        );
        color: #F3F7FC;
    }

    [data-testid="stSidebar"] {
        background: #091321;
        border-right: 1px solid #1E3348;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #12243A, #0D1B2D);
        border: 1px solid #243B52;
        padding: 20px;
        border-radius: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #A8BBD0;
    }

    [data-testid="stMetricValue"] {
        color: #54E1C1;
    }

    .hero {
        padding: 30px;
        border: 1px solid #24445B;
        border-radius: 22px;
        background: linear-gradient(
            120deg, #102A40, #102038 65%, #153A43
        );
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
        color: #FFFFFF;
    }

    .hero p {
        color: #B8CADB;
        font-size: 16px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #F4F8FC;
        margin: 20px 0 12px 0;
    }

    .hospital-card {
        background: #102238;
        border: 1px solid #28465D;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    .hospital-card h3 {
        color: #56E3C1;
        font-size: 18px;
    }

    .hospital-card p {
        color: #C0D0DF;
        font-size: 13px;
    }

    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #123D3A;
        color: #65E9CC;
        font-size: 12px;
        font-weight: 700;
    }

    .developer-card {
        padding: 16px;
        border: 1px solid #28465D;
        border-radius: 14px;
        background: #102238;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .developer-name {
        color: #56E3C1;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .developer-role {
        color: #B8CADB;
        font-size: 12px;
    }

    .footer {
        text-align: center;
        color: #A8BBD0;
        font-size: 12px;
        padding: 18px 0 8px 0;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #18BFA3, #3CD9B8);
        color: #06151E;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button:hover {
        background: #6BECD0;
        color: #06151E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD PROJECT FILES
# =========================================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


@st.cache_data
def load_results(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_test_data(path, modified_time):
    return pd.read_csv(path)


missing_files = [
    path.name
    for path in [MODEL_PATH, RESULTS_PATH, TEST_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "Required project files are missing: "
        + ", ".join(missing_files)
    )
    st.info(
        "Keep 3_app.py, federated_model.pkl, results.json, "
        "and global_test.csv in the same folder."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH), MODEL_PATH.stat().st_mtime
    )
    results = load_results(
        str(RESULTS_PATH), RESULTS_PATH.stat().st_mtime
    )
    test_df = load_test_data(
        str(TEST_PATH), TEST_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load the project files: {exc}")
    st.stop()


# =========================================================
# VALIDATE DATA
# =========================================================

if TARGET not in test_df.columns:
    st.error(f"The test dataset must contain a '{TARGET}' column.")
    st.stop()

feature_cols = [
    col for col in test_df.columns if col != TARGET
]

try:
    X_test = test_df[feature_cols].apply(
        pd.to_numeric, errors="raise"
    )
    y_test = test_df[TARGET]
except Exception as exc:
    st.error(f"Invalid dataset format: {exc}")
    st.stop()

if X_test.empty:
    st.error("The test dataset contains no feature rows.")
    st.stop()

if X_test.isna().any().any():
    st.error("The test dataset contains missing feature values.")
    st.stop()

if not np.isfinite(X_test.to_numpy(dtype=float)).all():
    st.error("The test dataset contains infinite values.")
    st.stop()

if hasattr(model, "n_features_in_"):
    if len(feature_cols) != model.n_features_in_:
        st.error(
            f"Feature mismatch: model expects "
            f"{model.n_features_in_} features, but the test "
            f"dataset contains {len(feature_cols)}."
        )
        st.stop()

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_cols:
        st.error(
            "The feature names or order differ from those used "
            "during training. Use the original training feature order."
        )
        st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_value(metrics, *keys):
    """Find a metric using any supported key name."""
    for key in keys:
        if key in metrics:
            try:
                return float(metrics[key])
            except (ValueError, TypeError):
                pass
    return None


def format_metric(value):
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def safe_label(value):
    """Display known class labels or the original class value."""
    try:
        numeric_value = int(value)
        return LABELS.get(numeric_value, f"Class {value}")
    except (ValueError, TypeError):
        return str(value)


def predict_details(row):
    """Predict using the same ordered features as the trained model."""
    row_df = pd.DataFrame([row], columns=feature_cols)

    prediction = model.predict(row_df)[0]
    classes = list(getattr(model, "classes_", []))

    confidence = None
    positive_probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row_df)[0]
        class_probabilities = dict(zip(classes, probabilities))

        if prediction in class_probabilities:
            confidence = float(class_probabilities[prediction])

        if 1 in class_probabilities:
            positive_probability = float(class_probabilities[1])

    label = safe_label(prediction)

    return prediction, label, confidence, positive_probability


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 🧬 FedCare AI")
    st.caption("Federated Intelligence Platform")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "AI Prediction",
            "Federated Training",
            "Model Evaluation",
        ],
    )

    st.divider()

    st.markdown("### System Status")
    st.markdown(
        '<span class="status-pill">● MODEL LOADED</span>',
        unsafe_allow_html=True,
    )

    st.write("**Participating hospitals:** 3")
    st.write(f"**Model features:** {len(feature_cols)}")
    st.write(f"**Test samples:** {len(test_df)}")

    st.caption(
        "Academic prototype • Simulated federated environment"
    )

    # Developer information
    st.divider()
    st.markdown("### Developed by")

    st.markdown(
        f"""
        <div class="developer-card">
            <div class="developer-name">{DEVELOPER_NAME}</div>
            <div class="developer-role">{DEVELOPER_ROLE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HERO HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <span class="status-pill">FEDERATED AI · FEDAVG</span>
        <h1>FedCare AI</h1>
        <p>
            Privacy-conscious collaborative learning for
            breast cancer prediction — powered by Federated Averaging.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown("### Platform Overview")

    fed_metrics = results.get("federated", {})

    accuracy = metric_value(
        fed_metrics, "Accuracy", "accuracy"
    )
    recall = metric_value(
        fed_metrics, "Recall", "recall"
    )
    f1 = metric_value(
        fed_metrics, "F1", "F1-score", "f1"
    )
    auc = metric_value(
        fed_metrics, "AUC", "ROC-AUC", "roc_auc"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Global Accuracy", format_metric(accuracy))
    c2.metric("Recall", format_metric(recall))
    c3.metric("F1-Score", format_metric(f1))
    c4.metric("ROC-AUC", format_metric(auc))

    st.markdown(
        '<div class="section-title">Federated Network</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    hospitals_info = [
        (col1, "🏥", "Hospital 01", "Local dataset & training"),
        (col2, "🏥", "Hospital 02", "Local dataset & training"),
        (col3, "🏥", "Hospital 03", "Local dataset & training"),
    ]

    for col, icon, title, description in hospitals_info:
        with col:
            st.markdown(
                f"""
                <div class="hospital-card">
                    <div style="font-size:34px">{icon}</div>
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <span class="status-pill">DATA STAYS LOCAL</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="text-align:center; padding:12px; color:#56E3C1;">
            ↓ &nbsp; Local model updates — not raw patient records &nbsp; ↓
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hospital-card">
            <h3>⚙️ Central Aggregation Server</h3>
            <p>
                FedAvg combines local model parameters into a global model.
            </p>
            <span class="status-pill">GLOBAL MODEL UPDATE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">How It Works</div>',
        unsafe_allow_html=True,
    )

    step1, step2, step3 = st.columns(3)

    step1.info(
        "**01 · Local Training**\n\n"
        "Each hospital trains the model using its own local data."
    )
    step2.info(
        "**02 · FedAvg Aggregation**\n\n"
        "The server computes a sample-weighted average of local parameters."
    )
    step3.info(
        "**03 · Global Update**\n\n"
        "The updated model is evaluated and can be sent back to clients."
    )

    history = results.get("history", [])

    if history:
        history_df = pd.DataFrame(history)

        if (
            "Round" in history_df.columns
            and "Accuracy" in history_df.columns
        ):
            st.markdown(
                '<div class="section-title">'
                'Global Model Learning Curve</div>',
                unsafe_allow_html=True,
            )

            fig = px.line(
                history_df,
                x="Round",
                y="Accuracy",
                markers=True,
                template="plotly_dark",
                title="Accuracy Across Federated Rounds",
            )

            fig.update_traces(line=dict(width=3))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_tickformat=".0%",
            )

            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# AI PREDICTION PAGE
# =========================================================

elif page == "AI Prediction":

    st.markdown("### Breast Cancer Prediction")
    st.write(
        "Choose a sample from the held-out test dataset, "
        "or enter feature values manually."
    )

    mode = st.radio(
        "Input mode",
        ["Test Dataset Sample", "Manual Feature Entry"],
        horizontal=True,
    )

    if mode == "Test Dataset Sample":

        sample_index = st.selectbox(
            "Select test sample",
            options=list(range(len(X_test))),
            format_func=lambda i: f"Test Sample #{i + 1}",
        )

        sample = X_test.iloc[sample_index].to_dict()

        with st.expander("Inspect input features"):
            st.dataframe(
                pd.DataFrame({
                    "Feature": feature_cols,
                    "Value": [sample[f] for f in feature_cols],
                }),
                use_container_width=True,
                hide_index=True,
            )

        actual = y_test.iloc[sample_index]

        if st.button("Run AI Prediction", use_container_width=True):

            prediction, label, confidence, positive_probability = (
                predict_details(sample)
            )

            if prediction == 1:
                st.error(f"Model Prediction: {label}")
            else:
                st.success(f"Model Prediction: {label}")

            left, right = st.columns(2)

            left.metric(
                "Model Confidence",
                format_metric(confidence),
            )

            right.metric(
                "Actual Dataset Label",
                safe_label(actual),
            )

            if positive_probability is not None:
                st.progress(
                    min(max(positive_probability, 0.0), 1.0)
                )
                st.caption(
                    "Displayed score for class 1. It is a model "
                    "probability when predict_proba is available."
                )

            if prediction == actual:
                st.success(
                    "Prediction matches the dataset label."
                )
            else:
                st.warning(
                    "Prediction differs from the dataset label."
                )

    else:

        st.warning(
            "Manual predictions must use the same feature units, "
            "feature order, and preprocessing as model training."
        )

        manual_values = {}

        with st.form("manual_prediction_form"):

            for start in range(0, len(feature_cols), 2):
                columns = st.columns(2)

                for j, col in enumerate(
                    feature_cols[start:start + 2]
                ):
                    values = X_test[col].astype(float)
                    median = float(values.median())
                    minimum = float(values.min())
                    maximum = float(values.max())

                    with columns[j]:
                        manual_values[col] = st.number_input(
                            col,
                            min_value=min(minimum, median),
                            max_value=max(maximum, median),
                            value=median,
                            format="%.5f",
                            key=f"feature_{col}",
                        )

            submitted = st.form_submit_button(
                "Analyze Feature Profile",
                use_container_width=True,
            )

        if submitted:

            prediction, label, confidence, positive_probability = (
                predict_details(manual_values)
            )

            if prediction == 1:
                st.error(f"Model Prediction: {label}")
            else:
                st.success(f"Model Prediction: {label}")

            st.metric(
                "Model Confidence",
                format_metric(confidence),
            )

            st.caption(
                "Experimental model output only — not a medical "
                "diagnosis. Clinical use requires independent validation."
            )


# =========================================================
# FEDERATED TRAINING PAGE
# =========================================================

elif page == "Federated Training":

    st.markdown("### Federated Learning Monitor")
    st.write(
        "Explore the saved training history from the FedAvg experiment."
    )

    history = results.get("history", [])

    if not history:
        st.warning(
            "No training history was found in results.json. "
            "Run the federated training script first."
        )

    else:
        history_df = pd.DataFrame(history)

        metric_options = [
            col
            for col in ["Accuracy", "Recall", "F1", "AUC"]
            if col in history_df.columns
        ]

        if metric_options and "Round" in history_df.columns:

            selected_metrics = st.multiselect(
                "Metrics to display",
                metric_options,
                default=metric_options,
            )

            if selected_metrics:
                chart_df = history_df[
                    ["Round"] + selected_metrics
                ].melt(
                    id_vars="Round",
                    var_name="Metric",
                    value_name="Score",
                )

                fig = px.line(
                    chart_df,
                    x="Round",
                    y="Score",
                    color="Metric",
                    markers=True,
                    template="plotly_dark",
                    title="Performance by Federated Round",
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_tickformat=".0%",
                )

                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    config = results.get("config", {})

    st.markdown("### Experiment Configuration")

    cfg1, cfg2, cfg3 = st.columns(3)

    cfg1.metric("Hospitals", config.get("hospitals", 3))
    cfg2.metric("Training Rounds", config.get("rounds", "N/A"))
    cfg3.metric("Local Epochs", config.get("local_epochs", "N/A"))

    st.caption(
        "This history comes from results.json. It is not a live "
        "connection to hospital servers."
    )


# =========================================================
# MODEL EVALUATION PAGE
# =========================================================

elif page == "Model Evaluation":

    st.markdown("### Model Performance Comparison")
    st.write(
        "Compare the local models, centralized reference, "
        "and federated model using the saved experiment results."
    )

    comparison_rows = []

    for hospital, metrics in results.get("local", {}).items():
        comparison_rows.append({
            "Model": hospital,
            "Accuracy": metric_value(
                metrics, "Accuracy", "accuracy"
            ),
            "Recall": metric_value(
                metrics, "Recall", "recall"
            ),
            "F1-Score": metric_value(
                metrics, "F1", "F1-score", "f1"
            ),
            "ROC-AUC": metric_value(
                metrics, "AUC", "ROC-AUC", "roc_auc"
            ),
        })

    central_metrics = results.get("centralized", {})

    if central_metrics:
        comparison_rows.append({
            "Model": "Centralized Reference",
            "Accuracy": metric_value(
                central_metrics, "Accuracy", "accuracy"
            ),
            "Recall": metric_value(
                central_metrics, "Recall", "recall"
            ),
            "F1-Score": metric_value(
                central_metrics, "F1", "F1-score", "f1"
            ),
            "ROC-AUC": metric_value(
                central_metrics, "AUC", "ROC-AUC", "roc_auc"
            ),
        })

    fed_metrics = results.get("federated", {})

    if fed_metrics:
        comparison_rows.append({
            "Model": "Federated (FedAvg)",
            "Accuracy": metric_value(
                fed_metrics, "Accuracy", "accuracy"
            ),
            "Recall": metric_value(
                fed_metrics, "Recall", "recall"
            ),
            "F1-Score": metric_value(
                fed_metrics, "F1", "F1-score", "f1"
            ),
            "ROC-AUC": metric_value(
                fed_metrics, "AUC", "ROC-AUC", "roc_auc"
            ),
        })

    if comparison_rows:

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                    "ROC-AUC": "{:.2%}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )

        plot_metric = st.selectbox(
            "Compare models by",
            ["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
        )

        plot_df = comparison_df.dropna(subset=[plot_metric])

        if not plot_df.empty:
            fig = px.bar(
                plot_df,
                x="Model",
                y=plot_metric,
                color="Model",
                text=plot_df[plot_metric].map(
                    lambda x: f"{x:.1%}"
                ),
                template="plotly_dark",
                title=f"{plot_metric} Comparison",
            )

            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis_tickformat=".0%",
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "No comparison metrics were found in results.json."
        )

    st.markdown("### Important Notes")

    st.info(
        "The centralized model is a reference baseline that uses pooled "
        "training data. The federated approach trains on local partitions "
        "and aggregates model parameters. Compare metrics before drawing "
        "conclusions about which approach performs better."
    )


# =========================================================
# FOOTER & DEVELOPER CREDIT
# =========================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        <strong>{DEVELOPER_NAME}</strong><br>
        {DEVELOPER_ROLE}<br><br>
        FedCare AI | Academic Federated Learning Prototype<br>
        © 2026 {DEVELOPER_NAME}. All rights reserved.<br>
        Not intended for clinical diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
