from pathlib import Path
import zipfile

out_dir = Path("/mnt/data/streamlit_churn_app")
out_dir.mkdir(parents=True, exist_ok=True)

app_code = r'''from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Customer Churn Prediction — Streamlit UI
# ============================================================
# Place app.py, IT_customer_churn.csv, and best_fraud_model.pkl
# in the same project folder before running/deploying.
#
# IMPORTANT:
# The categorical encoding below is reconstructed from the CSV.
# It MUST match the preprocessing used when the model was trained.
# For production, save and load the original preprocessing pipeline.
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

# -------------------------- Styling --------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      .stApp {
        background: #f5f7fa;
        color: #17212f;
      }
      [data-testid="stSidebar"] {
        background: #101923;
      }
      [data-testid="stSidebar"] * {
        color: #eef4f8;
      }
      .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #101923 0%, #1c3440 100%);
        color: white;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255,255,255,.08);
      }
      .hero-kicker {
        color: #9ce8c0;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .14em;
      }
      .hero h1 {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.6rem);
        line-height: 1.15;
        margin: .55rem 0;
        color: #ffffff;
      }
      .hero p {
        color: #c7d4dc;
        max-width: 720px;
        margin-bottom: 0;
      }
      .section-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        color: #17212f;
        font-size: 1.12rem;
        margin: .3rem 0 .75rem;
      }
      .metric-card {
        background: white;
        border: 1px solid #e4eaf0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 108px;
      }
      .metric-label {
        color: #667587;
        font-size: .82rem;
        margin-bottom: .35rem;
      }
      .metric-value {
        font-family: 'Manrope', sans-serif;
        color: #17212f;
        font-size: 1.35rem;
        font-weight: 800;
      }
      .result-card {
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #dce7e1;
        background: #ffffff;
      }
      .result-yes {
        border-left: 5px solid #d97757;
      }
      .result-no {
        border-left: 5px solid #2f9d72;
      }
      div.stButton > button, div.stFormSubmitButton > button {
        background: #187b5a;
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 2.8rem;
        font-weight: 700;
      }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #126247;
        color: white;
      }
      .footnote {
        color: #687789;
        font-size: .82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------- Load resources -----------------------
@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place it beside app.py."
        )
    # Only load trusted model files.
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading dataset schema...")
def load_dataset():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {DATASET_PATH.name}. "
            "Place it beside app.py."
        )

    data = pd.read_csv(DATASET_PATH)

    required = {
        "Churn", "customerID", "gender", "SeniorCitizen",
        "tenure", "MonthlyCharges", "TotalCharges",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.dropna(subset=["TotalCharges"]).copy()

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
        load_dataset()
    )
    load_error = None
except Exception as exc:
    model = None
    df = None
    FEATURE_COLUMNS = []
    NUMERIC_COLUMNS = []
    CATEGORICAL_COLUMNS = []
    encoders = {}
    load_error = str(exc)


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## ◈ ChurnLens")
    st.caption("CUSTOMER INTELLIGENCE")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("• Prediction workspace")
    st.markdown("• Model status")
    st.markdown("---")
    st.markdown("**Model**")
    if load_error:
        st.error("Model resources unavailable")
    else:
        st.success("Model resources loaded")
        st.caption(f"{len(FEATURE_COLUMNS)} input features")
    st.markdown("---")
    st.caption("Built with Streamlit · Pandas · XGBoost")


# ------------------------------ Header ------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Customer analytics / ML workspace</div>
      <h1>Customer Churn Prediction</h1>
      <p>Assess customer churn risk using account, service and billing details.
      Enter a customer profile to request a prediction from your trained model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"Unable to initialize the app: {load_error}")
    st.info(
        "Check that app.py, best_fraud_model.pkl, and "
        "IT_customer_churn.csv are in the same directory."
    )
    st.stop()

# Check feature count and, when available, feature order.
model_feature_count = getattr(model, "n_features_in_", None)
if (
    model_feature_count is not None
    and model_feature_count != len(FEATURE_COLUMNS)
):
    st.error(
        f"Model/schema mismatch: model expects {model_feature_count} "
        f"features but the dataset schema contains {len(FEATURE_COLUMNS)}. "
        "Use the exact feature preprocessing from model training."
    )
    st.stop()

model_feature_names = getattr(model, "feature_names_in_", None)
if model_feature_names is not None and list(model_feature_names) != FEATURE_COLUMNS:
    st.error(
        "Model feature names/order do not match the dataset feature order. "
        "Update the preprocessing to match the original training pipeline."
    )
    st.stop()

# KPI strip
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Dataset records</div>'
        f'<div class="metric-value">{len(df):,}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    churn_rate = (df["Churn"].astype(str).str.lower() == "yes").mean() * 100
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Historical churn rate</div>'
        f'<div class="metric-value">{churn_rate:.1f}%</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Model input features</div>'
        f'<div class="metric-value">{len(FEATURE_COLUMNS)}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
st.caption("Fields marked with options use categories found in the Telco churn dataset.")

# ------------------------------ Form ------------------------------
with st.form("customer_churn_form"):
    tab_account, tab_services, tab_billing = st.tabs(
        ["Account profile", "Services", "Billing"]
    )

    with tab_account:
        a1, a2, a3 = st.columns(3)
        with a1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
        with a2:
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
        with a3:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=72,
                value=12, step=1,
            )

    with tab_services:
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
        with s2:
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
        with s3:
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

    with tab_billing:
        b1, b2, b3 = st.columns(3)
        with b1:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox(
                "Paperless billing", ["Yes", "No"]
            )
        with b2:
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0,
                max_value=10000.0, value=70.0, step=1.0,
            )
        with b3:
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0,
                max_value=1000000.0, value=840.0, step=10.0,
            )

    submitted = st.form_submit_button(
        "Run churn prediction", use_container_width=True
    )


# ------------------------------ Prediction ------------------------------
if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        input_df = pd.DataFrame(
            [[customer[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

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

        # Assumption: training target used 1/Yes for churn.
        positive_labels = (1, True, "1", "Yes", "yes")
        is_churn = prediction in positive_labels
        churn_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            positive_index = next(
                (
                    idx for idx, label in enumerate(classes)
                    if label in positive_labels
                ),
                None,
            )
            if positive_index is not None:
                churn_probability = float(probabilities[positive_index])

        st.write("")
        st.markdown(
            '<div class="section-title">Prediction result</div>',
            unsafe_allow_html=True,
        )

        result_class = "result-yes" if is_churn else "result-no"
        result_title = "Churn risk detected" if is_churn else "Customer likely to stay"
        result_message = (
            "The model classified this customer as likely to churn."
            if is_churn
            else "The model classified this customer as likely to stay."
        )

        st.markdown(
            f'<div class="result-card {result_class}">'
            f'<h3 style="margin:0 0 .4rem;color:#17212f;">{result_title}</h3>'
            f'<p style="margin:0;color:#667587;">{result_message}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        with r1:
            st.metric("Predicted churn", "Yes" if is_churn else "No")
        with r2:
            if churn_probability is not None:
                st.metric("Churn probability", f"{churn_probability:.1%}")
                st.progress(min(max(churn_probability, 0.0), 1.0))
            else:
                st.metric("Churn probability", "Not available")

        st.caption(
            "This is a model estimate, not a guarantee of future customer behaviour."
        )

    except Exception as exc:
        st.error("Prediction failed. Check the model preprocessing and feature schema.")
        with st.expander("Technical details"):
            st.code(str(exc))


st.markdown("---")
st.markdown(
    '<div class="footnote">ChurnLens · Customer analytics demo · '
    'Verify model preprocessing and class labels before production use.</div>',
    unsafe_allow_html=True,
)
'''

requirements = """streamlit
pandas
numpy
scikit-learn
xgboost
joblib
"""

readme = """# ChurnLens Streamlit app

## Files required
Keep these files in the same folder:
- app.py
- IT_customer_churn.csv
- best_fraud_model.pkl

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Upload `app.py`, `requirements.txt`, the CSV, and model file to a GitHub repository.
2. Create a new app at https://share.streamlit.io/
3. Select the repository and set the main file path to `app.py`.
4. Deploy.

## Important model note
The app rebuilds LabelEncoder mappings from the CSV. This is only correct if the model was trained using the exact same mappings and feature order. Ideally, save the original preprocessing pipeline with the model and use that pipeline at inference time. Verify whether the target classes use 1/Yes to represent churn. The current app assumes that mapping.
"""

(out_dir / "app.py").write_text(app_code, encoding="utf-8")
(out_dir / "requirements.txt").write_text(requirements, encoding="utf-8")
(out_dir / "README.md").write_text(readme, encoding="utf-8")

zip_path = Path("/mnt/data/churnlens_streamlit_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for filename in ["app.py", "requirements.txt", "README.md"]:
        zf.write(out_dir / filename, arcname=filename)

print(f"Created: {out_dir / 'app.py'}")
print(f"Created: {out_dir / 'requirements.txt'}")
print(f"Created: {zip_path}")
