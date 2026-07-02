# ❤️ Explainable Heart Attack Risk Analyzer

A research-based healthcare machine learning project that analyzes real-world public health data, estimates heart attack risk, and explains model predictions using Explainable AI.

This project was built during my summer break after reading multiple research papers on heart disease prediction, healthcare machine learning, class imbalance, and explainable AI.

Heart attack is not a new problem. It has been one of the major healthcare challenges for a long time. But early risk awareness, responsible prediction, and explainable decision-making are still important real-world problems.

---

## 🚀 Project Overview

The goal of this project is to build an ML-based risk estimation system that can:

- Analyze large-scale public health data
- Understand heart attack risk patterns
- Ask users simple health-related questions
- Predict estimated heart attack risk probability
- Explain why the model predicted that risk
- Provide risk-reduction guidance responsibly

This is not a medical diagnosis system.  
It is a research-based risk estimation and analysis tool built using public health data.

---

## 📊 Dataset

The project uses CDC BRFSS multi-year public health survey data.

### Dataset Summary

| Metric | Value |
|---|---:|
| Total Records Processed | 4,847,365 |
| No Heart Attack Records | 4,567,805 |
| Heart Attack Records | 279,560 |
| Positive Class Ratio | ~5.77% |
| Model ROC-AUC | ~0.84 |
| Heart Attack Class Recall | ~87% |

The dataset is highly imbalanced, which makes this problem more realistic and challenging.

---

## 🧠 Problem Statement

In healthcare machine learning, accuracy alone can be misleading.

Since only around 5.77% of records belong to the heart attack class, a model can achieve high accuracy by predicting most users as “No Heart Attack.” But such a model may miss actual high-risk cases.

Because of this, the project focuses on:

- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Threshold tuning
- Explainability

---

## ⚙️ Features Used by the Model

The model uses health, lifestyle, and previous disease-related factors such as:

- Age group
- BMI
- Smoking status
- Physical activity
- Sleep hours
- Diabetes
- Stroke history
- Kidney disease
- COPD
- High blood pressure
- High cholesterol
- General health
- Routine checkup history
- Healthcare access-related factors

---

## 🤖 Machine Learning Approach

This project follows a complete real-world ML pipeline:

1. Research paper reading and problem analysis
2. Data collection
3. Data preprocessing
4. Missing value handling
5. Feature selection
6. Feature engineering
7. Class imbalance handling
8. Supervised ML model training
9. Model evaluation
10. Threshold tuning
11. SHAP explainability
12. Streamlit dashboard development

The trained model takes user inputs, converts them into a feature vector, predicts heart attack risk probability, applies a tuned threshold, and displays a risk category with explanation.

---

## 🔍 Explainability with SHAP

SHAP was used to make the model more transparent.

Instead of keeping the model as a black-box system, SHAP helps explain:

- Which features increased the estimated risk
- Which features reduced the estimated risk
- How much each feature influenced the prediction
- Global feature importance across the dataset

This makes the system more understandable and responsible for a healthcare-related use case.

---

## 🖥️ Dashboard Features

The project includes an interactive Streamlit dashboard with:

### 📊 Dataset Overview
Shows total records, heart attack cases, no heart attack cases, and yearly analysis.

### 📈 Interactive Graph Analysis
Allows users to analyze heart attack patterns by different factors like age, BMI, smoking, diabetes, high blood pressure, and more.

### 🫀 Risk Prediction Form
Users answer simple health-related questions, and the model estimates heart attack risk.

### 🔍 Explainability Page
Displays SHAP-based model explanations and feature importance.

### ⚠️ Medical Disclaimer
Clearly mentions that this project is not a diagnosis system and should not replace medical advice.

---

## 📌 Model Performance

| Metric | Value |
|---|---:|
| ROC-AUC | ~0.84 |
| Heart Attack Recall | ~87% |
| Positive Class Ratio | ~5.77% |

The model was designed as a risk-screening system, so recall and explainability were given more importance than accuracy alone.

---

## 🧩 Challenges Faced

Some major challenges during this project:

- Handling 48 lakh+ records
- Cleaning messy healthcare survey data
- Dealing with highly imbalanced classes
- Understanding why accuracy can be misleading
- Training large-scale ML models
- Python and SHAP compatibility issues
- Making the dashboard more product-like
- Writing healthcare-related output responsibly without claiming diagnosis or cure

---

## 📚 What I Learned

This project helped me learn:

- How to read research papers and convert ideas into a real project
- How real-world healthcare data is different from clean demo datasets
- How to handle large-scale CSV data
- Why class imbalance matters in healthcare ML
- Why recall, F1-score, ROC-AUC, and threshold tuning are important
- How SHAP explains model predictions
- How to build an interactive ML dashboard using Streamlit
- How to think about responsible AI in healthcare

Real-world ML is not just:

```python
model.fit()
