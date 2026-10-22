
# from pathlib import Path
# from typing import Literal

# import joblib
# import numpy as np
# import pandas as pd

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, ConfigDict, Field
# from sklearn.preprocessing import LabelEncoder

# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# # --------------------------------------------------
# # 1. File paths
# # --------------------------------------------------

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = BASE_DIR / "best_fraud_model.pkl"
# DATASET_PATH = BASE_DIR / "IT_customer_churn.csv"


# # --------------------------------------------------
# # 2. FastAPI application
# # --------------------------------------------------

# app = FastAPI(
#     title="Customer Churn Prediction API",
#     description=(
#         "Predict whether a customer is likely to churn "
#         "using an XGBoost classification model."
#     ),
#     version="1.0.0",
# )

# # Serve CSS and JavaScript files.
# app.mount(
#     "/static",
#     StaticFiles(directory=BASE_DIR / "static"),
#     name="static",
# )

# # Open the frontend at the root URL.
# @app.get("/", include_in_schema=False)
# def frontend():
#     return FileResponse(BASE_DIR / "static" / "index.html")
# # --------------------------------------------------
# # 3. Pydantic input validation
# # --------------------------------------------------

# class CustomerInput(BaseModel):
#     """Validate one customer's input data."""

#     model_config = ConfigDict(extra="forbid")

#     gender: Literal["Female", "Male"]
#     SeniorCitizen: int = Field(ge=0, le=1)

#     Partner: Literal["Yes", "No"]
#     Dependents: Literal["Yes", "No"]

#     tenure: int = Field(ge=0, le=72)

#     PhoneService: Literal["Yes", "No"]

#     MultipleLines: Literal[
#         "Yes", "No", "No phone service"
#     ]

#     InternetService: Literal[
#         "DSL", "Fiber optic", "No"
#     ]

#     OnlineSecurity: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     OnlineBackup: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     DeviceProtection: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     TechSupport: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     StreamingTV: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     StreamingMovies: Literal[
#         "Yes", "No", "No internet service"
#     ]

#     Contract: Literal[
#         "Month-to-month", "One year", "Two year"
#     ]

#     PaperlessBilling: Literal["Yes", "No"]

#     PaymentMethod: Literal[
#         "Electronic check",
#         "Mailed check",
#         "Bank transfer (automatic)",
#         "Credit card (automatic)",
#     ]

#     MonthlyCharges: float = Field(ge=0, le=10000)
#     TotalCharges: float = Field(ge=0, le=1000000)


# # --------------------------------------------------
# # 4. Load trained model and dataset
# # --------------------------------------------------

# if not MODEL_PATH.exists():
#     raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# if not DATASET_PATH.exists():
#     raise FileNotFoundError(
#         f"Dataset not found: {DATASET_PATH}"
#     )

# # Only load pickle files that you trust.
# model = joblib.load(MODEL_PATH)

# df = pd.read_csv(DATASET_PATH)

# # Clean numeric TotalCharges values.
# df["TotalCharges"] = pd.to_numeric(
#     df["TotalCharges"], errors="coerce"
# )

# df = df.dropna(subset=["TotalCharges"]).copy()

# # Target column is not an input feature.
# FEATURE_COLUMNS = [
#     column for column in df.columns
#     if column != "Churn"
# ]

# # These are categorical input columns.
# CATEGORICAL_COLUMNS = [
#     column for column in FEATURE_COLUMNS
#     if column not in [
#         "SeniorCitizen",
#         "tenure",
#         "MonthlyCharges",
#         "TotalCharges",
#     ]
# ]

# # Build category encoders from the dataset.
# # IMPORTANT: These must match the model's training
# # preprocessing exactly.
# encoders = {}

# for column in CATEGORICAL_COLUMNS:
#     encoder = LabelEncoder()
#     encoder.fit(df[column].astype(str))
#     encoders[column] = encoder


# # --------------------------------------------------
# # 5. API endpoints
# # --------------------------------------------------

# @app.get("/")
# def home():
#     return {
#         "message": "Customer Churn Prediction API is running",
#         "docs": "/docs",
#         "prediction_endpoint": "POST /predict",
#     }


# @app.get("/health")
# def health():
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "expected_features": len(FEATURE_COLUMNS),
#     }


# @app.post("/predict")
# def predict_churn(customer: CustomerInput):
#     try:
#         # Convert validated JSON into a DataFrame.
#         input_data = customer.model_dump()

#         input_df = pd.DataFrame(
#             [input_data],
#             columns=FEATURE_COLUMNS,
#         )

#         # Encode categorical features.
#         for column in CATEGORICAL_COLUMNS:
#             encoder = encoders[column]
#             value = str(input_df.at[0, column])

#             if value not in encoder.classes_:
#                 raise HTTPException(
#                     status_code=422,
#                     detail=f"Unsupported value for {column}: {value}",
#                 )

#             input_df[column] = encoder.transform(
#                 input_df[column].astype(str)
#             )

#         # Match the feature count expected by the model.
#         expected_count = getattr(
#             model, "n_features_in_", len(FEATURE_COLUMNS)
#         )

#         if len(FEATURE_COLUMNS) != expected_count:
#             raise HTTPException(
#                 status_code=500,
#                 detail=(
#                     f"Feature mismatch: API has "
#                     f"{len(FEATURE_COLUMNS)} features, "
#                     f"model expects {expected_count}."
#                 ),
#             )

#         # Make prediction.
#         prediction = int(model.predict(input_df)[0])

#         # Optional probability output.
#         probability = None

#         if hasattr(model, "predict_proba"):
#             probabilities = model.predict_proba(input_df)[0]
#             classes = list(model.classes_)

#             if 1 in classes:
#                 probability = round(
#                     float(probabilities[classes.index(1)]),
#                     4,
#                 )

#         # Assumes the training target was encoded:
#         # No -> 0 and Yes -> 1.
#         churn_label = "Yes" if prediction == 1 else "No"

#         return {
#             "prediction": prediction,
#             "churn": churn_label,
#             "churn_probability": probability,
#             "message": (
#                 "Customer is likely to churn."
#                 if prediction == 1
#                 else "Customer is likely to stay."
#             ),
#         }

#     except HTTPException:
#         raise

#     except Exception as exc:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(exc)}",
#         ) from exc
# ````````````````````````````````````````````````````````````````````````````````````````````````````````

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.preprocessing import LabelEncoder


# ==================================================
# 1. FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_fraud_model.pkl"
DATASET_PATH = BASE_DIR / "IT_customer_churn.csv"
STATIC_DIR = BASE_DIR / "static"


# ==================================================
# 2. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction with input validation.",
    version="1.0.0",
)

# Ensure frontend files exist before serving the application.
if not STATIC_DIR.is_dir():
    raise FileNotFoundError(f"Static folder not found: {STATIC_DIR}")

if not (STATIC_DIR / "index.html").is_file():
    raise FileNotFoundError(
        f"Frontend index.html not found in: {STATIC_DIR}"
    )

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ==================================================
# 3. PYDANTIC INPUT VALIDATION
# ==================================================

class CustomerInput(BaseModel):
    """Validate incoming customer data."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0, le=72)

    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal[
        "Yes", "No", "No phone service"
    ]

    InternetService: Literal[
        "DSL", "Fiber optic", "No"
    ]

    OnlineSecurity: Literal[
        "Yes", "No", "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes", "No", "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes", "No", "No internet service"
    ]

    TechSupport: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingTV: Literal[
        "Yes", "No", "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes", "No", "No internet service"
    ]

    Contract: Literal[
        "Month-to-month", "One year", "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(ge=0, le=10000)
    TotalCharges: float = Field(ge=0, le=1000000)


# ==================================================
# 4. LOAD MODEL AND DATASET
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# Load only trusted pickle/joblib files.
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATASET_PATH)

required_columns = {
    "Churn",
    "gender",
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: "
        f"{sorted(missing_columns)}"
    )

# Clean TotalCharges, which may contain blank strings.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["TotalCharges"]).copy()

# Exclude target and customer identifier from model inputs.
# customerID is an identifier, not a customer behaviour feature.
FEATURE_COLUMNS = [
    col for col in df.columns
    if col not in {"Churn", "customerID"}
]

# Validate the exact input schema.
expected_input_columns = set(CustomerInput.model_fields)
dataset_input_columns = set(FEATURE_COLUMNS)

if expected_input_columns != dataset_input_columns:
    raise ValueError(
        "Input schema does not match dataset features. "
        f"Missing from request model: "
        f"{sorted(dataset_input_columns - expected_input_columns)}. "
        f"Unexpected request fields: "
        f"{sorted(expected_input_columns - dataset_input_columns)}."
    )

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    col for col in FEATURE_COLUMNS
    if col not in NUMERIC_COLUMNS
]

# Label encoding is valid ONLY if it matches the training process.
encoders = {}

for column in CATEGORICAL_COLUMNS:
    encoder = LabelEncoder()
    encoder.fit(df[column].dropna().astype(str))
    encoders[column] = encoder

# Check the feature count before serving predictions.
expected_count = getattr(model, "n_features_in_", None)

if expected_count is not None and expected_count != len(FEATURE_COLUMNS):
    raise ValueError(
        f"Model expects {expected_count} features, "
        f"but dataset supplies {len(FEATURE_COLUMNS)}. "
        "Check the training preprocessing."
    )

# If the model stores feature names, check their order too.
model_feature_names = getattr(model, "feature_names_in_", None)

if model_feature_names is not None:
    if list(model_feature_names) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature names/order differ from the dataset. "
            "Use the exact training feature schema."
        )


# ==================================================
# 5. FRONTEND ROUTE
# ==================================================

@app.get("/", include_in_schema=False)
def frontend():
    """Serve the HTML frontend at the root URL."""
    return FileResponse(STATIC_DIR / "index.html")


# ==================================================
# 6. HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(FEATURE_COLUMNS),
    }


# ==================================================
# 7. CHURN PREDICTION
# ==================================================

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        # Convert validated input into the exact dataset order.
        input_data = customer.model_dump()

        input_df = pd.DataFrame(
            [[input_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        # Encode categorical values using dataset categories.
        for column in CATEGORICAL_COLUMNS:
            value = str(input_df.at[0, column])
            encoder = encoders[column]

            if value not in encoder.classes_:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported value for {column}: {value}",
                )

            input_df[column] = encoder.transform(
                input_df[column].astype(str)
            )

        # Make prediction.
        raw_prediction = model.predict(input_df)[0]

        # Support common target encodings:
        # 1 / "1" / True / "Yes" indicate churn.
        is_churn = raw_prediction in (1, True, "1", "Yes", "yes")

        churn_label = "Yes" if is_churn else "No"

        # Probability of the churn class, when available.
        churn_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            churn_class_index = next(
                (
                    i for i, label in enumerate(classes)
                    if label in (1, True, "1", "Yes", "yes")
                ),
                None,
            )

            if churn_class_index is not None:
                churn_probability = round(
                    float(probabilities[churn_class_index]),
                    4,
                )

        return {
            "prediction": churn_label,
            "churn": churn_label,
            "churn_probability": churn_probability,
            "message": (
                "Customer is likely to churn."
                if is_churn
                else "Customer is likely to stay."
            ),
        }

    except HTTPException:
        raise

    except Exception:
        # Log detailed errors in your server logs in production.
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check the model's training "
                "preprocessing and server logs."
            ),
        )
