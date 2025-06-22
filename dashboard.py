# # dashboard.py
# import streamlit as st
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from fraud_detection import (
#     load_data, preprocess_data, apply_pca, train_models, evaluate_models
# )

# st.title("💳 Credit Card Fraud Detection Dashboard")

# with st.spinner("Loading data..."):
#     df = load_data()
#     st.write("Raw Dataset (first 100 rows):")
#     st.dataframe(df.head(100))

# X, y = preprocess_data(df)

# pca_option = st.checkbox("Apply PCA", value=True)
# if pca_option:
#     n_components = st.slider("Number of PCA Components", 2, 20, 10)
#     X, _ = apply_pca(X, n_components)

# # Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # Train models
# models = train_models(X_train, y_train)
# results = evaluate_models(models, X_test, y_test)

# # Display ROC Curves
# st.subheader("📈 ROC Curves & AUC")
# fig, ax = plt.subplots()
# for name, res in results.items():
#     ax.plot(res['fpr'], res['tpr'], label=f"{name} (AUC={res['auc']:.4f})")
# ax.plot([0, 1], [0, 1], 'k--')
# ax.set_xlabel('False Positive Rate')
# ax.set_ylabel('True Positive Rate')
# ax.legend()
# st.pyplot(fig)

# # Classification Reports
# st.subheader("📋 Classification Reports")
# for name, res in results.items():
#     st.markdown(f"### {name}")
#     st.json(res['report'])




import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
from fraud_detection import load_data, preprocess_data, apply_pca, train_models, evaluate_models

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection")

# --- Load and Display Data ---
df = load_data()
st.sidebar.header("Data Info")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("📄 Raw Data Sample")
    st.dataframe(df.sample(100))

# --- Dataset Overview ---
st.sidebar.markdown("### Dataset Overview")
st.write("### 📈 Class Distribution")
fraud_count = df['Class'].value_counts()
st.bar_chart(fraud_count)

# --- Correlation Heatmap ---
if st.sidebar.checkbox("Show Correlation Heatmap"):
    st.subheader("🔍 Feature Correlation (PCA input excluded)")
    fig_corr, ax_corr = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.drop(columns=['Time']).corr(), cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

# --- Preprocessing ---
X, y = preprocess_data(df)

# --- PCA Option ---
apply_pca_option = st.sidebar.checkbox("Apply PCA", value=True)
if apply_pca_option:
    n_components = st.sidebar.slider("PCA Components", 2, 20, 10)
    X, _ = apply_pca(X, n_components)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Train Models ---
st.sidebar.markdown("### Train & Evaluate Models")
models = train_models(X_train, y_train)
results = evaluate_models(models, X_test, y_test)

# --- Model Selection ---
model_name = st.sidebar.selectbox("Choose a Model", list(results.keys()))
selected_result = results[model_name]

# --- ROC Curve ---
st.subheader(f"📈 ROC Curve – {model_name}")
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(selected_result["fpr"], selected_result["tpr"], label=f"AUC = {selected_result['auc']:.3f}")
ax_roc.plot([0, 1], [0, 1], 'k--')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend()
st.pyplot(fig_roc)

# --- Confusion Matrix ---
st.subheader(f"🧩 Confusion Matrix – {model_name}")
y_pred = models[model_name].predict(X_test)
fig_cm, ax_cm = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax_cm)
st.pyplot(fig_cm)

# --- Precision-Recall Curve ---
st.subheader(f"🎯 Precision-Recall Curve – {model_name}")
y_scores = models[model_name].predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)
fig_pr, ax_pr = plt.subplots()
ax_pr.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend()
st.pyplot(fig_pr)

# --- Classification Report ---
st.subheader(f"📋 Classification Report – {model_name}")
st.json(selected_result["report"])

# --- Download JSON Report ---
st.download_button(
    label="📥 Download Classification Report (JSON)",
    data=json.dumps(selected_result["report"], indent=4),
    file_name=f"{model_name}_report.json",
    mime="application/json"
)
