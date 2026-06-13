# 💳 Credit Card Fraud Detection System

## 📌 Overview

The Credit Card Fraud Detection System is a Machine Learning-powered web application that identifies fraudulent credit card transactions in real time. The system uses an XGBoost model for prediction, SHAP for explainable AI, and Django for the web interface.

Users can analyze transactions, view fraud probability, risk scores, risk levels, analytics dashboards, and generate downloadable reports.

---

## 🚀 Live Demo

**Deployed Application:**

https://credit-card-fraud-detection-sssm.onrender.com

---

## ✨ Features

### 🔍 Fraud Detection

* Predicts whether a transaction is Fraudulent or Legitimate
* Real-time transaction analysis
* Fraud probability calculation

### 📊 Risk Assessment

* Fraud Probability (%)
* Risk Score (0–100)
* Risk Level Classification:

  * Low
  * Medium
  * High
  * Critical

### 🤖 Explainable AI

* SHAP-based feature importance
* Displays top features influencing predictions
* Improves transparency and interpretability

### 📈 Analytics Dashboard

* Total Predictions
* Fraud Transactions Count
* Legitimate Transactions Count
* Average Risk Score
* Fraud Distribution Visualization
* Risk Level Distribution Charts

### 📄 Reporting

* Export prediction history to CSV
* Generate downloadable PDF reports

### 🌐 Deployment

* Hosted on Render
* Publicly accessible web application

---

## 🧠 Machine Learning Model

* XGBoost Classifier
* StandardScaler for feature scaling
* SHAP (SHapley Additive Explanations) for explainability

---

## 🛠️ Technology Stack

### Backend

* Python
* Django

### Machine Learning

* Scikit-learn
* XGBoost
* SHAP

### Frontend

* HTML
* CSS
* Bootstrap 5

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Reporting

* ReportLab
* CSV Export

### Deployment

* GitHub
* Render

---

## 📂 Project Structure


creditcardfraud/
│
├── fraud_app/
├── fraud_detection/
├── models/
│   ├── xgboost_model.pkl
│   └── scaler.pkl
├── static/
├── templates/
├── requirements.txt
├── build.sh
├── runtime.txt
├── manage.py
└── README.md


---

## 📊 Dataset

Credit Card Fraud Detection Dataset

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains anonymized PCA-transformed transaction features (V1–V28), transaction time, amount, and fraud labels.

---

## ⚙️ Installation

Clone the repository:


git clone <https://github.com/Vansh2708/creditcardfrauddetection.git>
cd creditcardfraud


Install dependencies:


pip install -r requirements.txt

Run migrations:


python manage.py migrate

Start the server:


python manage.py runserver

Open:


http://127.0.0.1:8000/

---

## 📋 Project Workflow

1. User enters transaction details.
2. Features are scaled using StandardScaler.
3. XGBoost model predicts transaction status.
4. Fraud probability is calculated.
5. Risk score and risk level are assigned.
6. SHAP explains the prediction.
7. Results are stored in prediction history.
8. Dashboard visualizes fraud analytics.
9. Reports can be downloaded as PDF or CSV.

---

## 🔮 Future Enhancements

* User Authentication
* Real-Time Transaction API Integration
* Email Fraud Alerts
* Advanced Analytics Dashboard
* Deep Learning-Based Fraud Detection
* Cloud Database Integration

---

## 👨‍💻 Author

**Vansh Jain**

B.Tech Computer Science Student

Machine Learning | Artificial Intelligence | Full Stack Development

---

## ⭐ Project Highlights

* End-to-End Machine Learning Deployment
* Explainable AI using SHAP
* Interactive Django Dashboard
* PDF & CSV Reporting
* Cloud Deployment on Render
* Professional UI using Bootstrap
