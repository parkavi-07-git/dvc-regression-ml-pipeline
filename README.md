
# DVC-Based Regression Machine Learning Pipeline

## 📌 Project Overview

This project implements an end-to-end Machine Learning regression pipeline using DVC (Data Version Control).

The project uses the Diabetes dataset available through Scikit-learn and predicts a continuous target value using a Random Forest Regression model.

DVC is used to manage the machine learning pipeline, dependencies, parameters, outputs, and reproducibility.

---

## 🔄 Machine Learning Pipeline

Data Ingestion
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Regression Metrics

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- DVC
- Git
- GitHub
- PyYAML
- Joblib

---

## 📂 Project Structure

```text
dvc-regression-ml-pipeline/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
├── model.pkl
├── requirements.txt
├── .gitignore
├── .dvcignore
└── README.md
