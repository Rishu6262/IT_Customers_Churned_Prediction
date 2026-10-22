
from pathlib import Path
import joblib
import xgboost

model_path = Path("best_fraud_model.pkl")

print("XGBoost version:", xgboost.__version__)
print("Model file exists:", model_path.exists())

if model_path.exists():
    print("Model file size:", model_path.stat().st_size, "bytes")

    try:
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        print("Model type:", type(model))

    except Exception as error:
        print("Model loading failed:", repr(error))
