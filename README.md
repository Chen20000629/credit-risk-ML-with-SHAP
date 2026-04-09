# 💳 Credit Risk Prediction with Explainable AI (SHAP)

## Project Overview
This project builds a machine learning model to predict loan default risk using the Kaggle "Give Me Some Credit" dataset.

The project focuses not only on prediction performance, but also on **model interpretability** using SHAP.

---

## Objective
- Predict whether a borrower will default
- Improve risk detection (focus on Recall)
- Explain model decisions for real-world applicability

---

## Models Used
- Logistic Regression (baseline)
- Random Forest (main model)

---

## Evaluation Metrics
- ROC-AUC
- Precision / Recall (focus on Recall for risk detection)

---

## Model Explainability (SHAP)

### 🔹 Feature Importance
![SHAP Summary](outputs/shap_summary.png)

- Debt ratio and past delinquency are key drivers of default risk
- High values of these features significantly increase risk

---

### 🔹 Individual Prediction Explanation
![SHAP Waterfall](outputs/shap_waterfall.png)

- Demonstrates how each feature contributes to a single prediction
- Helps interpret why a borrower is classified as high risk

---

## ROC Curve
![ROC Curve](outputs/roc_curve.png)

---

## Tech Stack
- Python
- Pandas
- Scikit-learn
- SHAP
- Matplotlib

---

## Key Takeaways
- Random Forest outperforms Logistic Regression in recall
- SHAP provides transparency for model decisions
- Suitable for real-world risk management scenarios

---

## Project Structure