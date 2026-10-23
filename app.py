from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


# ============================================================
# ChurnLens — Customer Churn Prediction
# ============================================================
# Required files in the same folder:
#   app.py
#   best_fraud_model.pkl
#   IT_customer_churn.csv
#
# IMPORTANT:
# LabelEncoder mappings below are reconstructed from the CSV.
# They must match the model's training preprocessing exactly.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_fraud_model.pkl"
DATASET_PATH = BASE_DIR / "IT_customer_churn.csv"

st.set_page_config(
    page_title="ChurnLens | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- Theme -------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #F4F7FA; color: #17212F; }
    [data-testid="stSidebar"] { background: #101923; }
    [data-testid="stSidebar"] * { color: #EAF0F5; }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #101923 0%, #1D3542 100%);
        color: #FFFFFF;
        border: 1px solid #263846;
        margin-bottom: 1.35rem;
    }
    .hero-kicker {
        color: #9CE8C0; font-size: .76rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: .16em;
    }
    .hero h1 {
        font-family: 'Manrope', sans-serif; color: #FFFFFF;
        font-size: clamp(1.9rem, 3vw, 2.65rem);
        font-weight: 800; margin: .55rem 0 .45rem;
    }
    .hero p { color: #CBD6DF; max-width: 760px; margin: 0; line-height: 1.65; }
    h4 { color: #17212F; font-family: "Manrope", sans-serif; font-weight: 800; margin-top: 1rem; }
    [data-testid="stForm"] { background: #FFFFFF; border: 1px solid #E1E8EF; border-radius: 16px; padding: 1.2rem 1.3rem; }
    [data-testid="stWidgetLabel"] p { color: #344256; font-weight: 600; }


    .metric-card {
        background: #FFFFFF; border: 1px solid #E1E8EF;
        border-radius: 15px; padding: 1.1rem 1.2rem; min-height: 106px;
    }
    .metric-label { color: #687789; font-size: .82rem; margin-bottom: .4rem; }
    .metric-value {
        font-family: 'Manrope', sans-serif; color: #17212F;
        font-size: 1.45rem; font-weight: 800;
    }
    .risk-panel {
        background: #101923; color: #F8FAFC; border: 1px solid #2A3948;
        border-radius: 18px; padding: 1.55rem 1.65rem; margin: .7rem 0 1rem;
    }
    .risk-muted { color: #A8B5C3; font-size: .88rem; }
    .risk-title { color: #F8FAFC; font-size: 1.35rem; font-weight: 800; margin-top: .25rem; }
    .risk-number { font-family: 'Manrope', sans-serif; font-size: 2.35rem; font-weight: 800; line-height: 1.2; }
    .risk-track {
        height: 13px; width: 100%; background: #303B48;
        border-radius: 99px; overflow: hidden; margin-top: .7rem;
    }
    .risk-fill { height: 100%; border-radius: 99px; }
    .risk-legend { display: flex; align-items: center; gap: .5rem; color: #D7E0E8; font-size: .83rem; }
    .risk-dot { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }
    .result-copy { color: #D7E0E8; font-size: .9rem; margin-top: 1rem; line-height: 1.55; }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #187B5A; color: #FFFFFF; border: 0;
        border-radius: 10px; min-height: 2.85rem; font-weight: 700;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #126247; color: #FFFFFF;
    }
    .footnote { color: #687789; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Load data -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    # Only load pickle files from a trusted source.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset_and_encoders():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH.name}")

    data = pd.read_csv(DATASET_PATH)
    data.columns = data.columns.astype(str).str.strip()

    required = {
        "Churn", "gender", "SeniorCitizen",
        "tenure", "MonthlyCharges", "TotalCharges",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset missing columns: {sorted(missing)}. "
            f"Available columns: {data.columns.tolist()}"
        )

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).copy()

    # Identifier is excluded if present; it is not a predictive feature.
    feature_columns = [
        col for col in data.columns
        if col not in {"Churn", "customerID"}
    ]

    numeric_columns = [
        col for col in [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
        ] if col in feature_columns
    ]
    categorical_columns = [
        col for col in feature_columns if col not in numeric_columns
    ]

    encoders = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit(data[column].dropna().astype(str))
        encoders[column] = encoder

    return data, feature_columns, numeric_columns, categorical_columns, encoders


try:
    model = load_model()
    df, FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, encoders = (
        load_dataset_and_encoders()
    )
except Exception as exc:
    st.error(f"App initialization failed: {exc}")
    st.info(
        "Ensure app.py, best_fraud_model.pkl, and "
        "IT_customer_churn.csv are in the repository root."
    )
    st.stop()


# ----------------------- Model/schema validation -----------------------
model_count = getattr(model, "n_features_in_", None)
if model_count is not None and model_count != len(FEATURE_COLUMNS):
    st.error(
        f"Feature mismatch: model expects {model_count} features, "
        f"but dataset schema provides {len(FEATURE_COLUMNS)}. "
        "Use the exact preprocessing and feature list used during training."
    )
    st.stop()

model_names = getattr(model, "feature_names_in_", None)
if model_names is not None and list(model_names) != FEATURE_COLUMNS:
    st.error(
        "Model feature names/order do not match the dataset. "
        "Use the original training feature order."
    )
    st.stop()


# ------------------------------- Sidebar -------------------------------
with st.sidebar:
    st.markdown("## ◈ ChurnLens")
    st.caption("CUSTOMER INTELLIGENCE")
    st.markdown("---")
    st.markdown("**Workspace**")
    st.markdown("Customer prediction")
    st.markdown("Risk assessment")
    st.markdown("---")
    st.success("Model and dataset loaded")
    st.caption(f"{len(FEATURE_COLUMNS)} input features")
    st.markdown("---")
    st.caption("Streamlit · Pandas · XGBoost")


# -------------------------------- Header --------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Customer analytics / ML workspace</div>
      <h1>Customer Churn Prediction</h1>
      <p>Evaluate a customer's likelihood of leaving using account,
      service and billing information. Complete the profile and run
      the trained classification model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

churn_rate = (
    df["Churn"].astype(str).str.strip().str.lower().eq("yes").mean() * 100
)

metric_columns = st.columns(3)
metrics = [
    ("Dataset records", f"{len(df):,}"),
    ("Historical churn rate", f"{churn_rate:.1f}%"),
    ("Model input features", str(len(FEATURE_COLUMNS))),
]
for column, (label, value) in zip(metric_columns, metrics):
    with column:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.subheader("Customer profile")
st.caption("Account, service and billing details are grouped for easier entry.")


# -------------------------------- Form --------------------------------
with st.form("customer_churn_form"):
    st.markdown("#### Account information")
    a1, a2, a3 = st.columns(3)
    with a1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox(
            "Senior citizen", [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        partner = st.selectbox("Has partner", ["Yes", "No"])
    with a2:
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72,
            value=12, step=1,
        )
        phone = st.selectbox("Phone service", ["Yes", "No"])
    with a3:
        lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )

    st.markdown("---")
    st.markdown("#### Internet and streaming services")
    s1, s2, s3 = st.columns(3)
    with s1:
        security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
    with s2:
        protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
    with s3:
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )

    st.markdown("---")
    st.markdown("#### Contract and billing")
    b1, b2, b3 = st.columns(3)
    with b1:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with b2:
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly charges ($)", min_value=0.0,
            max_value=10000.0, value=70.0, step=1.0,
        )
    with b3:
        total = st.number_input(
            "Total charges ($)", min_value=0.0,
            max_value=1000000.0, value=840.0, step=10.0,
        )

    st.caption("All 19 customer input features are shown above.")
    submitted = st.form_submit_button(
        "Run churn prediction", use_container_width=True
    )

# ----------------------------- Prediction -----------------------------
if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
    }

    try:
        # Build input in the dataset's feature order.
        input_df = pd.DataFrame(
            [[customer[column] for column in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        # This mapping must be identical to model-training preprocessing.
        for column in CATEGORICAL_COLUMNS:
            value = str(input_df.at[0, column])
            encoder = encoders[column]

            if value not in encoder.classes_:
                st.error(f"Unsupported category for {column}: {value}")
                st.stop()

            input_df[column] = encoder.transform(
                input_df[column].astype(str)
            )

        prediction = model.predict(input_df)[0]
        classes = list(getattr(model, "classes_", []))

        # Confirm the positive class from the model's actual class labels.
        positive_labels = (1, True, "1", "Yes", "yes")
        churn_class_index = next(
            (
                index for index, label in enumerate(classes)
                if label in positive_labels
            ),
            None,
        )

        if churn_class_index is None:
            st.error(
                f"Cannot identify the churn class from model labels: {classes}. "
                "Check how the target was encoded during training."
            )
            st.stop()

        is_churn = prediction in positive_labels
        churn_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            churn_probability = float(probabilities[churn_class_index])

        st.write("")
        st.subheader("Prediction result")

        if churn_probability is not None:
            risk_pct = max(0.0, min(100.0, churn_probability * 100))
            remaining_pct = 100.0 - risk_pct

            if risk_pct >= 70:
                risk_label = "High risk"
                risk_color = "#E9A64B"
            elif risk_pct >= 40:
                risk_label = "Moderate risk"
                risk_color = "#60A5FA"
            else:
                risk_label = "Low risk"
                risk_color = "#34D399"

            result_text = (
                "The model predicts this customer may churn."
                if is_churn
                else "The model predicts this customer is likely to stay."
            )

            st.markdown(
                f"""
                <div class="risk-panel">
                  <div style="display:flex;justify-content:space-between;
                              align-items:flex-start;gap:20px;flex-wrap:wrap;">
                    <div>
                      <div class="risk-muted">CHURN RISK ASSESSMENT</div>
                      <div class="risk-title">{risk_label}</div>
                    </div>
                    <div style="text-align:right;">
                      <div class="risk-muted">Churn probability</div>
                      <div class="risk-number" style="color:{risk_color};">
                        {risk_pct:.1f}%
                      </div>
                    </div>
                  </div>

                  <div style="display:flex;justify-content:space-between;
                              gap:12px;margin-top:25px;flex-wrap:wrap;">
                    <div class="risk-legend">
                      <span class="risk-dot" style="background:{risk_color};"></span>
                      Churn risk <strong>{risk_pct:.1f}%</strong>
                    </div>
                    <div class="risk-legend">
                      <span class="risk-dot" style="background:#303B48;"></span>
                      Remaining <strong>{remaining_pct:.1f}%</strong>
                    </div>
                  </div>

                  <div class="risk-track">
                    <div class="risk-fill"
                         style="width:{risk_pct:.4f}%;background:{risk_color};">
                    </div>
                  </div>

                  <div class="result-copy">{result_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if is_churn:
                st.warning("Churn risk detected by the classification model.")
            else:
                st.success("The model predicts this customer is likely to stay.")
            st.info("This model does not provide predict_proba output.")

        st.caption(
            "A model probability is an estimate, not a guarantee of future behaviour. "
            "Validate calibration and training preprocessing before business use."
        )

    except Exception as exc:
        st.error("Prediction failed. Verify feature schema and preprocessing.")
        with st.expander("Technical details"):
            st.code(str(exc))


st.markdown("---")
st.markdown(
    '<div class="footnote">ChurnLens · Customer analytics workspace · '
    'Verify model provenance and preprocessing before production use.</div>',
    unsafe_allow_html=True,
)
