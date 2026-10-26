from pathlib import Path

 📊 ChurnLens — Customer Churn Prediction System

> An end-to-end Machine Learning application that estimates customer churn risk using customer account, service, contract, and billing information.

---

## 🚀 Live Demo

Try the deployed Streamlit application:

🌐 **Live Application:** https://it-customers-churned-prediction-p19.streamlit.app

The live application lets users enter a customer profile and request a churn prediction through a browser. The output depends on the saved model and preprocessing code deployed with the app.

> **Important:** A successful prediction response does not automatically mean the prediction is accurate. Before using this application for business decisions, verify that the saved model was trained for customer churn and that its preprocessing, feature order, and target-label mapping match the application.

---

## ✨ Features

- 📉 **Customer churn prediction** — predicts the class returned by the trained classification model.
- 🎯 **Churn probability display** — displays a model-estimated churn probability when the estimator supports `predict_proba()` and the churn class is identified correctly.
- 🚦 **Visual risk assessment** — presents a percentage bar and a Low, Moderate, or High risk label based on the UI thresholds.
- 👤 **Customer profile form** — collects account and demographic-style fields used by the model.
- 🌐 **Service information** — captures phone, internet, security, backup, device protection, support, and streaming selections.
- 💳 **Contract and billing inputs** — collects contract duration, payment method, paperless billing, monthly charges, and total charges.
- 🧪 **Dataset validation** — checks important columns and normalizes surrounding whitespace in column names.
- 🔍 **Model schema checks** — checks feature count and feature names/order when the estimator exposes that information.
- ⚡ **Interactive Streamlit UI** — allows a user to enter values and run a prediction without using a command-line interface.
- ☁️ **Cloud deployment** — the application is hosted using Streamlit Community Cloud.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

# 📌 1. Project Overview

## What is customer churn?

Customer churn means a customer stops using a company's product or service. For example, a telecom customer may cancel their internet plan, switch to another provider, or stop paying for a subscription.

A churn prediction system uses patterns learned from historical customer data to estimate whether a customer belongs to the churn or non-churn class. A business could use validated predictions to investigate customer experience problems or evaluate retention strategies.

## What does ChurnLens do?

ChurnLens is a web application built with Python and Streamlit. A user enters customer details into a form. The application converts those values into a tabular input, applies the preprocessing implemented in the app, and sends the prepared data to a saved classification model.

The application then displays:
1. The predicted class, such as `Yes` or `No`, depending on the model's target encoding.
2. An estimated churn probability, if the model supports probability prediction and the correct churn class can be identified.
3. A visual risk indicator to make the probability easier to interpret.

## Why build this project?

This project demonstrates how a trained Machine Learning model can be integrated into a usable application. Instead of leaving the model inside a notebook, the project provides a form that can be accessed through a browser.

It brings together data loading, feature preparation, categorical encoding, model loading, prediction, user-interface design, and cloud deployment.

---

# 🎯 2. Project Objectives

| Objective | Detailed explanation |
|---|---|
| Understand churn data | Inspect the dataset and identify which columns describe a customer and which column represents the target. |
| Validate data | Confirm that essential columns exist and that numeric charge fields can be processed. |
| Prepare model inputs | Exclude the target column and any identifier that is not intended as a predictive feature. |
| Encode categories | Convert categorical values into the representation expected by the trained model. |
| Run inference | Load the saved model and use it to predict a customer record. |
| Display probability | Use `predict_proba()` only when supported and only after identifying the correct churn class. |
| Build an interface | Provide a structured form and clear result display through Streamlit. |
| Deploy the app | Host the app so it can be opened in a browser without running it locally. |
| Document limitations | Explain why model training, preprocessing, and target-label mapping must be verified. |

---

# 📂 3. Dataset Information

The application reads a CSV file named **`IT_customer_churn.csv`**.

| Attribute | Details |
|---|---|
| Dataset file | `IT_customer_churn.csv` |
| Target column | `Churn` |
| Data format | Structured tabular CSV |
| Problem type | Binary classification, assuming the target contains two churn classes |
| Domain | Customer analytics / churn prediction |
| Identifier | `customerID`, excluded from features if it exists |

## What is a target variable?

A target variable is the value a model is trained to predict. In this project, `Churn` is intended to represent whether a customer churned.

For example, a dataset might use `Yes` for customers who left and `No` for customers who stayed. The exact representation must be checked in the actual dataset and model-training code; this README does not assume that every saved model uses the same encoding.

## Why is `customerID` excluded?

A customer ID is usually an identifier, not a meaningful description of customer behaviour. Passing it into a model may cause the model to learn accidental patterns that do not generalize. The current app excludes `customerID` from the input features if the column is present.

## Data cleaning in the app

The application performs basic preparation before constructing encoders:

- It reads the CSV using Pandas.
- It removes leading and trailing whitespace from column names.
- It converts `TotalCharges` to numeric values where possible.
- It removes rows with missing `TotalCharges` before building categorical encoders.

These steps help avoid common CSV issues. They do not replace a full data-quality and model-training pipeline.

---

# 📋 4. Input Features — Detailed Explanation

The interface collects 19 input fields. The selected values must match the categories and formats used during model training.

| Feature | Meaning | Example values |
|---|---|---|
| `gender` | Customer gender category as represented in the dataset. | `Female`, `Male` |
| `SeniorCitizen` | Binary indicator for senior-citizen status. | `0`, `1` |
| `Partner` | Whether the customer has a partner. | `Yes`, `No` |
| `Dependents` | Whether the customer has dependents. | `Yes`, `No` |
| `tenure` | Number of months the customer has stayed with the provider. | `0`–`72` in the current form |
| `PhoneService` | Whether phone service is active. | `Yes`, `No` |
| `MultipleLines` | Whether the customer has multiple phone lines, or the relevant no-service category. | `Yes`, `No`, `No phone service` |
| `InternetService` | Type of internet service. | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Whether online security is included. | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | Whether online backup is included. | `Yes`, `No`, `No internet service` |
| `DeviceProtection` | Whether device protection is included. | `Yes`, `No`, `No internet service` |
| `TechSupport` | Whether technical support is included. | `Yes`, `No`, `No internet service` |
| `StreamingTV` | Whether streaming TV service is included. | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | Whether streaming movies service is included. | `Yes`, `No`, `No internet service` |
| `Contract` | Contract duration category. | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling` | Whether billing is paperless. | `Yes`, `No` |
| `PaymentMethod` | Payment method category. | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | Current monthly charge amount. | Non-negative number |
| `TotalCharges` | Total accumulated charge amount. | Non-negative number |

### Why do these fields matter?

These fields describe different aspects of a customer relationship: how long the customer has stayed, what services they use, which contract they have, and how they are billed. The model learns statistical relationships between these inputs and the target labels in the training data.

A feature's presence does not mean it causes churn. The model identifies patterns from its training data, and those patterns must be evaluated before being used for business decisions.

---

# 🛠 5. Technology Stack

| Technology | Role in this project | Detailed explanation |
|---|---|---|
| **Python** | Main programming language | Connects the data-loading, validation, preprocessing, prediction, and UI components. |
| **Pandas** | Data handling | Reads the CSV into a DataFrame and prepares columns for inference. |
| **scikit-learn** | Preprocessing interface | Provides `LabelEncoder` and common model interfaces used by the application. |
| **Joblib** | Model loading | Loads the serialized estimator from the `.pkl` file. |
| **XGBoost** | Optional model runtime | Required if the saved estimator depends on XGBoost. The actual estimator type must be confirmed. |
| **Streamlit** | Web interface | Creates form controls, sections, buttons, messages, and the result display in Python. |
| **Git and GitHub** | Version control | Store project files and track code changes. |
| **Streamlit Community Cloud** | Hosting | Runs the app online from a GitHub repository. |

## Python libraries

### Pandas
Pandas reads the dataset and organizes it as rows and columns. A DataFrame is useful because most tabular ML models expect input features in a structured format.

### scikit-learn
The application imports `LabelEncoder` to convert categorical strings into numeric labels. It also relies on common estimator attributes and methods, such as `predict()`, `predict_proba()`, `classes_`, and `n_features_in_`, when the saved model provides them.

### Joblib
Joblib loads the serialized model from disk. Serialized model files should only be loaded from trusted sources because pickle-based formats can execute code during loading.

### Streamlit
Streamlit turns Python code into an interactive web application. It provides form controls, layout columns, buttons, status messages, and result components without requiring a separate HTML/CSS/JavaScript frontend.

---

# ⚙️ 6. Project Workflow — Step by Step

## Step 1: Load the files

The application locates files relative to the directory containing `app.py`. This makes paths more reliable than hardcoding a machine-specific path such as `C:\Users\...`.

The app loads:
- `IT_customer_churn.csv`
- The serialized model referenced by `MODEL_PATH`

## Step 2: Validate the dataset

The app checks whether essential columns exist. If required columns are missing, it reports the missing column names rather than continuing with an incomplete schema.

Column names are stripped of surrounding whitespace so that a header such as `TotalCharges ` can be normalized to `TotalCharges`.

## Step 3: Prepare the feature list

The target column `Churn` is not a prediction input. It must be excluded because the model is supposed to predict it.

If `customerID` is present, it is also excluded. The remaining columns are used to build the model's input schema.

## Step 4: Separate numeric and categorical features

Numeric fields include:
- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

The other model-input columns are treated as categorical fields by the current implementation. This classification must match the training pipeline.

## Step 5: Build categorical encoders

The current app fits a `LabelEncoder` for each categorical column using categories from the CSV. This maps category strings to integers.

**Important limitation:** This is correct only if the category-to-integer mappings are identical to those used when training the model. If training used a different encoder or preprocessing pipeline, the current approach can produce invalid predictions. A saved, fitted preprocessing pipeline is the safer approach.

## Step 6: Check the model schema

When available, the app checks:
- `n_features_in_`: how many input features the estimator expects.
- `feature_names_in_`: feature names and their order recorded by the estimator.

A feature-count match alone does not guarantee correct preprocessing. Feature order and encoding must also match the training process.

## Step 7: Collect user inputs

The Streamlit form groups fields into logical sections:
- Account information
- Internet and streaming services
- Contract and billing

The form gathers 19 values and uses them to build a single-customer input row.

## Step 8: Create the input DataFrame

The selected values are placed into a Pandas DataFrame in the expected feature order. This matters because a model interprets each input position or named column as a particular feature.

## Step 9: Encode categorical inputs

Categorical values are converted to numeric values using the encoder objects. If an input category is not recognized, the app should report an error rather than silently invent a mapping.

## Step 10: Generate a prediction

The model's `predict()` method returns the predicted class. The application must verify which class label means churn by checking the model's classes and the target encoding used during training.

If the estimator supports `predict_proba()`, the app can retrieve probabilities for its classes. The probability at the correct churn-class index is the value that should be displayed as churn probability.

## Step 11: Display the result

The result section can show:
- Predicted churn class
- Estimated churn probability
- A visual probability bar
- A Low, Moderate, or High label based on UI thresholds

The risk label is a presentation rule, not proof that a particular probability range is objectively low or high risk. Those thresholds should be validated against the use case.

## 🔄 High-Level Architecture

```text
User opens the web app
          |
          v
Streamlit customer form
          |
          v
Input validation and DataFrame creation
          |
          v
Categorical preprocessing
          |
          v
Saved classification model
          |
          +------> Predicted class
          |
          +------> Class probabilities (if supported)
          |
          v
Churn result and risk visualization
```

---

# 📁 7. Project Structure

```text
churnlens/
│
├── app.py
├── requirements.txt
├── IT_customer_churn.csv
├── best_fraud_model.pkl
└── README.md
```

| File | Purpose |
|---|---|
| `app.py` | Contains the Streamlit interface, file loading, data checks, feature preparation, prediction, and result rendering. |
| `requirements.txt` | Lists Python packages that the local environment or deployment platform must install. |
| `IT_customer_churn.csv` | Dataset used by the current app to inspect the schema and reconstruct categorical encoder classes. |
| `best_fraud_model.pkl` | Serialized model file currently referenced by the app. Verify that it is a churn-trained model before use. |
| `README.md` | Project overview, setup guide, workflow explanation, limitations, and deployment instructions. |

> **Model filename warning:** `best_fraud_model.pkl` contains the word `fraud`. A filename does not establish the model's training target. Confirm the training script and target labels. If this file is a fraud-detection model, it must be replaced with the correctly trained churn model.

---

# 💻 8. Run the Project Locally

## Prerequisites

- Python version compatible with the model and its serialized libraries.
- pip package installer.
- The CSV and trusted model file.
- Project files in the expected locations.

## Step 1: Open the project folder

In a terminal, navigate to the folder containing `app.py`:

```bash
cd path/to/churnlens
```

Replace `path/to/churnlens` with your actual project folder.

## Step 2: Create a virtual environment

A virtual environment isolates this project's Python packages from other projects.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file should include Streamlit, Pandas, scikit-learn, Joblib, and XGBoost if the serialized model requires it. For reproducible deployment, use versions compatible with the training environment.

## Step 4: Run the application

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, commonly:

```text
http://localhost:8501
```

Open the URL in your browser, complete the customer form, and click the prediction button.

---

# ☁️ 9. Deploy on Streamlit Community Cloud

## Step 1: Prepare the GitHub repository

Make sure the repository contains the application, dependency file, dataset, model file, and README.

Before uploading data or a model publicly, confirm that you are authorized to share those files. Do not commit secrets, credentials, or private customer information.

## Step 2: Push the files to GitHub

Commit the updated files and push them to the repository branch you want to deploy.

## Step 3: Create the Streamlit app

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in and connect GitHub.
3. Select the repository and branch.
4. Set the main file path to `app.py`.
5. Click **Deploy**.

## Step 4: Review logs

If deployment fails, open the logs and find the first relevant error. Fix the underlying issue, push the correction to GitHub, and allow the app to rebuild.

A successful deployment means the app started; it does not independently verify model quality or prediction correctness.

---

# 🧠 10. Model, Preprocessing, and Probability — Important Details

## Why preprocessing must match training

A trained model expects inputs in the same representation it saw during training. If training encoded categories one way and inference encodes them differently, predictions may be wrong even if the app runs without errors.

For example, suppose training used:

```text
Female -> 0
Male   -> 1
```

If inference uses a different mapping, the model may interpret the input incorrectly. The same principle applies to feature order, missing-value handling, scaling, and one-hot encoding.

### Recommended production design

The more reliable approach is to fit preprocessing and the estimator together in a pipeline, save that fitted pipeline, and load it in the application. The app should pass raw values with the expected feature names to the saved pipeline. It should not fit new encoders at inference time.

## What does churn probability mean?

If the model supports `predict_proba()`, it returns an estimated probability for each class. The application must identify the correct class index for churn before displaying one value as churn probability.

For a binary model with correctly identified churn probability \(p\), the complementary class probability is \(1-p\). This arithmetic does not guarantee that the probabilities are calibrated or that they predict real-world churn frequency accurately.

## What do Low, Moderate, and High mean?

These labels are produced by UI thresholds. For example, the app may assign High to probabilities at or above a selected threshold. Unless those thresholds have been validated using evaluation data and business requirements, they are illustrative categories only.

## How should the model be verified?

Before relying on results:
1. Confirm the model was trained with `Churn` as the target.
2. Confirm the saved feature list and feature order.
3. Confirm the exact categorical preprocessing.
4. Confirm how the target labels map to churn and non-churn.
5. Evaluate the model on held-out data.
6. Test known examples through the original training pipeline and compare results.

---

# 🧰 11. Troubleshooting

## Dataset file not found

**Possible reason:** The CSV is missing, stored in another folder, or named differently.

**What to check:**
- Confirm `IT_customer_churn.csv` is next to `app.py`.
- Check spelling and capitalization.
- Update `DATASET_PATH` if the file is stored elsewhere.

## Model file not found

**Possible reason:** The saved model is absent or its filename differs from `MODEL_PATH`.

**What to check:**
- Confirm the expected model file exists.
- Check the exact spelling.
- Only load trusted Joblib/pickle files because loading an untrusted serialized file can execute malicious code.

## Dataset missing columns

**Possible reason:** The CSV uses different headers, or a column name contains whitespace.

**What to check:**
- Print `df.columns.tolist()` to inspect the actual headers.
- Strip leading and trailing spaces.
- Compare the dataset schema with the required columns in `app.py`.
- Do not remove required-column checks without confirming the model's actual schema.

## Feature count or order mismatch

**Possible reason:** The model was trained on a different feature list, includes an identifier, or uses another preprocessing pipeline.

**What to check:**
- Compare the training feature list with the inference feature list.
- Exclude the target and identifier consistently.
- Preserve the exact feature order.
- Do not arbitrarily remove features to silence the error.

## Prediction fails or looks incorrect

**Possible reason:** The model file, input encoding, target mapping, or feature order is incompatible.

**What to check:**
- Verify that the model is for churn prediction.
- Verify categorical and numeric preprocessing.
- Confirm the positive churn class.
- Test a known validation row through the original training pipeline.

## Dependency or XGBoost warning

**Possible reason:** The model was serialized using a different library version, or a required package is missing.

**What to check:**
- Use compatible Python and ML library versions.
- Add XGBoost to `requirements.txt` if required by the model.
- Follow the supported XGBoost serialization workflow when migrating an older model, then validate predictions.

---

# 💡 12. Why This Project Is Valuable

- **End-to-end development:** Connects data preparation, model inference, UI development, and deployment.
- **Practical ML integration:** Shows how a saved model can be used outside a notebook.
- **Data handling:** Demonstrates CSV loading, schema checks, and categorical encoding.
- **Application development:** Uses Streamlit to turn a model into an interactive tool.
- **Deployment practice:** Uses GitHub and Streamlit Community Cloud to publish an application.
- **Portfolio value:** Provides a base project that can be improved with stronger model validation, testing, and monitoring.

---

# 🚧 13. Future Improvements

1. **Save a fitted preprocessing pipeline** so inference uses the same transformations as training.
2. **Add model evaluation metrics**, including precision, recall, F1-score, ROC-AUC, and a confusion matrix.
3. **Calibrate probabilities** if probability estimates will inform decisions.
4. **Validate risk thresholds** using held-out data and the intended business use case.
5. **Add automated tests** for required fields, feature order, category handling, and prediction behaviour.
6. **Pin dependency versions** to make deployment more reproducible.
7. **Add batch prediction** for a CSV of customer records.
8. **Provide downloadable results** where appropriate and safe.
9. **Monitor data drift and performance** after deployment.
10. **Create a model card** documenting training data, evaluation, intended use, and limitations.

---

# ✅ 14. Conclusion

ChurnLens demonstrates how a saved classification model can be connected to an interactive Streamlit application for customer churn prediction. The project combines Python, Pandas, categorical preprocessing, model inference, a browser-based interface, and cloud deployment.

The project is a strong learning foundation, but trustworthy predictions depend on validating the training target, preprocessing pipeline, feature order, target-class mapping, and model performance. These checks should be completed before using the output for real customer or business decisions.

---

# 👨‍💻 Author

## Rishu Gurjar

🎓 **B.Tech Computer Science Engineering Student**

💻 **Python Developer | Data Analyst | Machine Learning Enthusiast**

I am interested in building practical applications using Python, Data Analytics, Machine Learning, and AI. I enjoy developing end-to-end projects that connect data processing and trained models with usable applications.

### 📬 Connect with Me

- **GitHub:** https://github.com/Rishu6262
- **LinkedIn:** https://www.linkedin.com/in/rishu-gurjar-58072a333/
- **Kaggle:** https://www.kaggle.com/rishugurjar

⭐ If you find this project useful, consider giving the repository a star and sharing feedback.
'''
