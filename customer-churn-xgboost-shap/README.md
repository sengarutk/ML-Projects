# 🔍 Customer Churn Prediction with XGBoost + SHAP

Predict customer churn using advanced machine learning models and explain predictions with SHAP.

## 📌 Overview
This project tackles a real-world business problem: predicting if a customer is likely to leave (churn) based on service usage patterns and demographics. It uses the powerful **XGBoost** model for classification, with **SHAP values** for model interpretability.

## 🚀 Features
- Clean and preprocess customer data (nulls, encoding, scaling)
- Train-test pipeline using XGBoost Classifier
- SHAP visualizations for global & local interpretability
- Evaluation metrics: Accuracy, ROC AUC, F1-Score
- Confusion matrix and classification report

## 🧠 Tech Stack
- Python
- XGBoost
- SHAP
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

## 📁 Dataset
**Telco Customer Churn Dataset**  
📦 Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

## ⚙️ Setup Instructions
```bash
git clone https://github.com/yourusername/customer-churn-xgboost-shap
cd customer-churn-xgboost-shap
pip install -r requirements.txt
python churn_prediction.py
