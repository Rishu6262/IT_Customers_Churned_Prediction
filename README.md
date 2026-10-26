📊 ChurnLens — Customer Churn Prediction System

---

## 🚀 Live Demo

Experience the ChurnLens customer churn prediction application.

🌐 **Live Application:**  
https://it-customers-churned-prediction-p19.streamlit.app

---

## ✨ Features

- 📉 Predict whether a customer may churn
- 🎯 Display estimated churn probability as a percentage
- 🚦 Show a visual churn-risk indicator
- 👤 Collect customer account information
- 🌐 Analyze internet and streaming service selections
- 💳 Include contract and billing information
- ⚡ Interactive machine learning predictions
- 📱 Browser-based Streamlit application
- 🧪 Validate dataset columns and model feature compatibility

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Project-brightgreen)

**ChurnLens** is an end-to-end Machine Learning project that predicts whether a telecom customer may leave a service based on account profile, subscribed services, contract type, and billing information. The project combines data preparation, classification, model serialization, and an interactive Streamlit interface.

---

# 📌 Project Overview

Customer churn occurs when a customer stops using a company's product or service. Predicting churn risk can help organizations identify customers who may need additional support or retention offers.

The ChurnLens application accepts customer information through a web form and passes the data to a saved classification model. The app displays the predicted churn class and, when the model supports `predict_proba()` and the churn class is correctly identified, an estimated churn probability.

The application is designed as a practical learning and portfolio project demonstrating the integration of Python, Pandas, scikit-learn-compatible preprocessing, a serialized machine learning model, and Streamlit deployment.

> **Model validation note:** Before relying on predictions, verify that the model file is trained for customer churn and that its feature order, categorical preprocessing, and target-label mapping match the application code.

---

# 🎯 Project Objectives

- 📂 Understand and prepare customer churn data.
- 🧹 Validate dataset columns and clean numeric charge values.
- 🔎 Identify the model input features and exclude the target column.
- ⚙️ Encode categorical input features for model inference.
- 🤖 Load a trained classification model and generate predictions.
- 📊 Display a churn probability when supported by the model.
- 🎨 Build a clean and interactive Streamlit UI.
- ☁️ Prepare the project for deployment on Streamlit Community Cloud.
- 💼 Demonstrate an end-to-end machine learning application workflow.

---

# 📂 Dataset Information

The application uses a CSV file named **`IT_customer_churn.csv`**.

| Attribute | Details |
|---|---|
| Dataset file | `IT_customer_churn.csv` |
| Target variable | `Churn` |
| Data type | Structured tabular data |
| Domain | Customer analytics / churn prediction |
| Identifier handling | `customerID` is excluded from model features if present |

## 📋 Input Features

The UI collects 19 customer features:

| Feature | Description |
|---|---|
| `gender` | Customer gender category |
| `SeniorCitizen` | Senior-citizen indicator (`0` or `1`) |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `PhoneService` | Whether phone service is enabled |
| `MultipleLines` | Multiple-line service category |
| `InternetService` | Internet service type |
| `OnlineSecurity` | Online security subscription |
| `OnlineBackup` | Online backup subscription |
| `DeviceProtection` | Device protection subscription |
| `TechSupport` | Tech support subscription |
| `StreamingTV` | Streaming TV subscription |
| `StreamingMovies` | Streaming movies subscription |
| `Contract` | Contract duration |
| `PaperlessBilling` | Paperless billing preference |
| `PaymentMethod` | Customer payment method |
| `MonthlyCharges` | Monthly charge amount |
| `TotalCharges` | Total accumulated charges |

The selectable values in the application should match the categories and preprocessing used during model training.

---

# 🛠 Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| Programming language | Python | Application logic |
| Data processing | Pandas | Read and prepare CSV data |
| Machine learning | scikit-learn-compatible estimator | Classification and encoding |
| Model serialization | Joblib | Load the saved model |
| Optional ML runtime | XGBoost | Required if the serialized model uses XGBoost |
| Web application | Streamlit | Interactive frontend and app hosting |
| Version control | Git & GitHub | Source control and deployment |

## 📚 Main Python Libraries

- **Pandas** — load and prepare dataset columns.
- **scikit-learn** — `LabelEncoder` and compatible ML interfaces.
- **Joblib** — load the trained model.
- **Streamlit** — create the interactive web application.
- **XGBoost** — include if the saved model depends on XGBoost.

---

# ⚙️ Project Workflow

The application follows this inference workflow:

### 1. Load the files
- Read `IT_customer_churn.csv`.
- Load the saved model from the project directory.

### 2. Validate the dataset
- Check for required columns.
- Strip whitespace from column names.
- Convert `TotalCharges` to numeric values where possible.
- Remove rows with missing `TotalCharges` for encoder preparation.

### 3. Prepare the feature schema
- Exclude `Churn`, the target column.
- Exclude `customerID` if present.
- Identify categorical and numeric features.
- Check the expected model feature count and feature names when available.

### 4. Collect customer input
- Enter account details.
- Select services and subscription options.
- Enter contract and billing information.

### 5. Encode and predict
- Convert categorical values using encoders reconstructed from the CSV.
- Arrange the fields in the expected feature order.
- Call the model's `predict()` method.
- Call `predict_proba()` if supported and use the correct churn class index.

### 6. Display the result
- Show the predicted churn class.
- Display estimated churn probability when available.
- Show a visual risk bar and a risk label.

## 🔄 High-Level Architecture

```text
Customer
   |
   v
Streamlit Input Form
   |
   v
Input DataFrame Preparation
   |
   v
Categorical Encoding
   |
   v
Trained Classification Model
   |
   +----> Predicted Churn Class
   |
   +----> Churn Probability (if supported)
   |
   v
Risk Assessment UI
```

---

# 📁 Project Structure

```text
churnlens/
│
├── app.py
├── requirements.txt
├── IT_customer_churn.csv
├── best_fraud_model.pkl
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit interface, data validation, preprocessing, prediction, and results |
| `requirements.txt` | Python package dependencies |
| `IT_customer_churn.csv` | Dataset used to inspect columns and reconstruct categorical encoders |
| `best_fraud_model.pkl` | Serialized model file referenced by the app |
| `README.md` | Project documentation |

**Important:** The current model filename contains the word `fraud`. Confirm from the model-training code that it is actually a churn model. If it is a fraud-detection model, replace it with the correctly trained churn model and update the filename in `app.py`.

---

When updating the app, commit and push your changes to GitHub. Streamlit Community Cloud can then rebuild the app.

> **Security:** Do not upload credentials, private customer information, or confidential data to a public repository. Only publish model and dataset files you are authorized to share.

---

# 🧠 Model and Preprocessing Notes

## Categorical encoding

The current application reconstructs `LabelEncoder` mappings from the CSV when the app starts. This approach is valid only if those category-to-number mappings and the feature order are identical to the preprocessing used when the model was trained.

For example, if the model was trained with `Female → 0` and `Male → 1`, inference must use those same mappings. If the training pipeline used one-hot encoding, scaling, or a fitted `ColumnTransformer`, reconstructing `LabelEncoder` instances will not reproduce that pipeline.

**Recommended improvement:** Save the complete fitted preprocessing-and-model pipeline and load it during inference. Pass raw, correctly named feature values to that pipeline rather than fitting new encoders from the dataset each time.

## Target class and probability

The app must identify which model class means churn. The current code expects a positive class such as `1`, `True`, `"1"`, `"Yes"`, or `"yes"`. Verify the model's `classes_` and the target encoding in the training script.

A `predict_proba()` value is a model-estimated probability, not a guarantee of customer behaviour. The UI's Low, Moderate, and High risk thresholds are illustrative and should be validated against model performance and business needs.

## Model file verification

The application currently points to `best_fraud_model.pkl`. A filename does not prove which target the model was trained to predict. Verify that the saved model was trained using customer churn labels before interpreting its output as churn risk.

---

# 🧰 Troubleshooting

### Dataset file not found
- Ensure `IT_customer_churn.csv` is in the same directory as `app.py`.
- Check spelling and capitalization.
- Update `DATASET_PATH` if the file is stored elsewhere.

### Model file not found
- Ensure the expected model file is beside `app.py`.
- Check the exact filename and update `MODEL_PATH` if needed.
- Load only trusted Joblib/pickle files.

### Missing dataset columns
- Inspect `df.columns.tolist()`.
- Check for whitespace or different column names.
- Compare the dataset schema with the required columns in `app.py`.

### Feature mismatch
- Compare the training feature list with the inference feature list.
- Keep the same column order and preprocessing as training.
- Do not remove or reorder features arbitrarily to silence an error.

### Incorrect predictions
- Confirm the saved model is trained for churn prediction.
- Verify category encoding and numeric preprocessing.
- Confirm the positive churn class mapping.
- Test a known validation row through the original training pipeline.

### Dependency errors
- Use Python and package versions compatible with the environment used to train the model.
- Add XGBoost to `requirements.txt` if the saved model depends on it.
- For older serialized XGBoost models, follow the supported XGBoost model serialization workflow and validate predictions after any migration.

---

# 💡 Why This Project Is Valuable

- Demonstrates an end-to-end machine learning inference workflow.
- Combines data handling, categorical preprocessing, and model integration.
- Uses a real-world customer analytics problem.
- Includes an interactive application instead of only a notebook.
- Provides experience with Streamlit deployment and GitHub version control.
- Creates a foundation for further work in model evaluation, monitoring, and responsible AI.

---

# 🚧 Future Improvements

- Save a complete fitted preprocessing-and-model pipeline.
- Add model evaluation metrics such as precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Calibrate probabilities and select risk thresholds using validation data.
- Add automated tests for input schema and prediction behaviour.
- Pin dependency versions for reproducible deployments.
- Add batch prediction and downloadable reports.
- Monitor data drift and model performance.
- Add a model card describing training data, intended use, and limitations.
- Add screenshots of the final deployed interface.

---

# ✅ Conclusion

ChurnLens demonstrates how a trained classification model can be connected to an interactive Streamlit application for customer churn prediction. It brings together Python, data preparation, categorical encoding, model inference, and web deployment in a single project.

Before using it for real business decisions, validate the model's training target, preprocessing pipeline, feature order, target-class mapping, and prediction quality.

---

# 👨‍💻 Author

## Rishu Gurjar

🎓 **B.Tech Computer Science Engineering Student**

💻 **Python Developer | Data Analyst | Machine Learning Enthusiast**

I am interested in developing practical applications using Python, Data Analytics, Machine Learning, and AI. I enjoy building end-to-end projects that connect data processing and trained models with usable applications.

### 📬 Connect with Me
- **GitHub:** https://github.com/Rishu6262
- **LinkedIn:** https://www.linkedin.com/in/rishu-gurjar-58072a333/
- **Kaggle:** https://www.kaggle.com/rishugurjar

⭐ If you find this project useful, consider giving the repository a star and sharing feedback.
