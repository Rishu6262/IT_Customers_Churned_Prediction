from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_fraud_model.pkl"
DATASET_PATH = BASE_DIR / "IT_customer_churn.csv"

st.set_page_config(
    page_title="ChurnLens | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#f5f7fa;color:#17212f;}
[data-testid="stSidebar"] {background:#101923;}
[data-testid="stSidebar"] * {color:#eef4f8;}
.hero {padding:1.5rem 1.7rem;border-radius:18px;background:linear-gradient(120deg,#101923,#1c3440);color:white;margin-bottom:1.25rem;}
.kicker {color:#9ce8c0;font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;}
.hero h1 {font-family:'Manrope',sans-serif;color:white;font-size:clamp(1.8rem,3vw,2.6rem);margin:.55rem 0;}
.hero p {color:#c7d4dc;max-width:720px;margin-bottom:0;}
.card {background:white;border:1px solid #e4eaf0;border-radius:14px;padding:1rem 1.1rem;min-height:100px;}
.label {color:#667587;font-size:.82rem;margin-bottom:.35rem;}
.value {font-family:'Manrope',sans-serif;color:#17212f;font-size:1.35rem;font-weight:800;}
.result {background:white;border:1px solid #dce7e1;border-radius:16px;padding:1.25rem;}
div.stButton>button,div.stFormSubmitButton>button {background:#187b5a;color:white;border:0;border-radius:10px;min-height:2.8rem;font-weight:700;}
div.stButton>button:hover,div.stFormSubmitButton>button:hover {background:#126247;color:white;}
.note {color:#687789;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    # Load only trusted model files.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset_and_encoders():
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH.name}")

    data = pd.read_csv(DATASET_PATH)
    required = {"Churn",
        "gender",
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).copy()

    features = [c for c in data.columns if c not in {"Churn", "customerID"}]
    numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [c for c in features if c not in numeric]

    encoders = {}
    for col in categorical:
        enc = LabelEncoder()
        enc.fit(data[col].dropna().astype(str))
        encoders[col] = enc

    return data, features, numeric, categorical, encoders


try:
    model = load_model()
    df, FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, encoders = (
        load_dataset_and_encoders()
    )
except Exception as exc:
    st.error(f"App initialization failed: {exc}")
    st.info("Place app.py, best_fraud_model.pkl, and IT_customer_churn.csv in the same folder.")
    st.stop()

model_count = getattr(model, "n_features_in_", None)
if model_count is not None and model_count != len(FEATURE_COLUMNS):
    st.error(
        f"Feature mismatch: model expects {model_count} features, "
        f"but dataset schema provides {len(FEATURE_COLUMNS)}."
    )
    st.stop()

model_names = getattr(model, "feature_names_in_", None)
if model_names is not None and list(model_names) != FEATURE_COLUMNS:
    st.error("Model feature names/order do not match the dataset feature order.")
    st.stop()

with st.sidebar:
    st.markdown("## ◈ ChurnLens")
    st.caption("CUSTOMER INTELLIGENCE")
    st.markdown("---")
    st.markdown("**Workspace**")
    st.markdown("Prediction")
    st.markdown("Model status")
    st.markdown("---")
    st.success("Model and dataset loaded")
    st.caption(f"{len(FEATURE_COLUMNS)} input features")

st.markdown("""
<div class="hero">
  <div class="kicker">Customer analytics / ML workspace</div>
  <h1>Customer Churn Prediction</h1>
  <p>Assess customer churn risk using account, service and billing details.
  Enter a customer profile to request a prediction from your trained model.</p>
</div>
""", unsafe_allow_html=True)

churn_rate = (df["Churn"].astype(str).str.lower() == "yes").mean() * 100
m1, m2, m3 = st.columns(3)
for col, label, value in [
    (m1, "Dataset records", f"{len(df):,}"),
    (m2, "Historical churn rate", f"{churn_rate:.1f}%"),
    (m3, "Model input features", str(len(FEATURE_COLUMNS))),
]:
    with col:
        st.markdown(
            f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
st.subheader("Customer profile")
st.caption("Enter account, service and billing details below.")

with st.form("customer_churn_form"):
    account_tab, services_tab, billing_tab = st.tabs(
        ["Account profile", "Services", "Billing"]
    )

    with account_tab:
        a1, a2, a3 = st.columns(3)
        with a1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
        with a2:
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
        with a3:
            tenure = st.number_input("Tenure (months)", 0, 72, 12, 1)

    with services_tab:
        s1, s2, s3 = st.columns(3)
        with s1:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        with s2:
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with s3:
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])

    with billing_tab:
        b1, b2, b3 = st.columns(3)
        with b1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        with b2:
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly = st.number_input("Monthly charges ($)", min_value=0.0,
                                      max_value=10000.0, value=70.0, step=1.0)
        with b3:
            total = st.number_input("Total charges ($)", min_value=0.0,
                                    max_value=1000000.0, value=840.0, step=10.0)

    submitted = st.form_submit_button("Run churn prediction", use_container_width=True)

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
        # Preserve the dataset/model feature order.
        input_df = pd.DataFrame(
            [[customer[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        # NOTE: This encoding must exactly match model training.
        for col in CATEGORICAL_COLUMNS:
            value = str(input_df.at[0, col])
            encoder = encoders[col]
            if value not in encoder.classes_:
                st.error(f"Unsupported category for {col}: {value}")
                st.stop()
            input_df[col] = encoder.transform(input_df[col].astype(str))

        prediction = model.predict(input_df)[0]
        classes = list(getattr(model, "classes_", []))
        positive_labels = (1, True, "1", "Yes", "yes")
        is_churn = prediction in positive_labels

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            positive_idx = next(
                (i for i, label in enumerate(classes)
                 if label in positive_labels),
                None,
            )
            if positive_idx is not None:
                probability = float(probabilities[positive_idx])

        st.write("")
        st.subheader("Prediction result")
        if is_churn:
            st.warning("Churn risk detected — the model predicts this customer may churn.")
        else:
            st.success("Customer likely to stay — the model predicts no churn.")

        r1, r2 = st.columns(2)
        r1.metric("Predicted churn", "Yes" if is_churn else "No")
        if probability is not None:
            r2.metric("Churn probability", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        else:
            r2.metric("Churn probability", "Not available")

        st.caption("Model output is an estimate, not a guarantee of future customer behaviour.")
    except Exception as exc:
        st.error("Prediction failed. Verify model preprocessing and feature schema.")
        with st.expander("Technical details"):
            st.code(str(exc))

st.markdown("---")
st.markdown(
    '<div class="note">ChurnLens · Customer analytics demo · '
    'Verify preprocessing and target labels before production use.</div>',
    unsafe_allow_html=True,
)
