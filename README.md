# 💳 Credit Card Fraud Detection

**Type**: Binary Classification (Anomaly Detection)**Tech Stack**: Python · Scikit-learn · XGBoost · Streamlit · PCA

---

## 📌 Overview

This project builds an end-to-end machine learning pipeline to detect fraudulent credit card transactions using the Kaggle dataset. It tackles the challenges of **highly imbalanced classes** and **real-world anomaly detection** by applying preprocessing, dimensionality reduction, robust models, and interpretable metrics.

---

## 🚀 Key Features

- 🔄 **Data Preprocessing**: Feature scaling, outlier handling, and PCA (Principal Component Analysis) for dimensionality reduction
- ⚖️ **Imbalanced Data Handling**: Class weights, ROC-AUC and PR curve evaluation
- 🧠 **Model Training & Comparison**: Logistic Regression, Random Forest, and XGBoost
- 📈 **Model Evaluation**: ROC-AUC, PR AUC, classification reports, confusion matrices
- 📊 **Interactive Dashboard**: Built with Streamlit to visualize performance metrics, compare models, and display fraud detection outcomes

---

## 📂 Project Structure

```
├── creditcard.csv                # Input dataset (Kaggle)
├── fraud_detection.py            # ML pipeline: load, preprocess, train, evaluate
├── dashboard.py                  # Streamlit app for visualization
├── requirements.txt              # Dependencies
└── README.md                     # Project summary
```

---

## 📉 Metrics Used

- **ROC-AUC**: Overall model separability
- **PR Curve**: Fraud detection quality under class imbalance
- **Precision/Recall**: Trade-off between false positives and missed frauds

---

## 📊 Model Results (Example)

| Model               | ROC AUC | Precision | Recall |
| ------------------- | ------- | --------- | ------ |
| Logistic Regression | 0.94    | 0.84      | 0.63   |
| Random Forest       | 0.96    | 0.87      | 0.69   |
| XGBoost             | 0.97    | 0.89      | 0.72   |

---

## ✅ Outcomes

- Realistic approach to fraud detection using industry-grade tools
- Highlights the challenges and solutions in working with **rare-event classification**
- Interactive tool for **explaining, comparing, and visualizing** fraud detection models

---

## 🛠️ How to Run

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle and place `creditcard.csv` in the root project folder from `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data`
4. Launch dashboard: `streamlit run dashboard.py`

---
