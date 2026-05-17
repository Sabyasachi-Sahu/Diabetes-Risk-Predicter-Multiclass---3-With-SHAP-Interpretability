# Diabetes Risk Predicter (Multiclass 3) With SHAP Interpretability

# 🏥 Diabetes Risk Assessment Tool
This repository contains an end-to-end Machine Learning project designed to predict the risk of diabetes based on lifestyle and health indicators. The project leverages a large dataset from CDC survey records to provide three classification results: No Diabetes, Prediabetes, and Diabetes.

## 📋 Project Overview
Diabetes is a major global health concern. This project aims to identify key risk factors and provide a user-friendly interface for individuals to assess their health status using historical data and predictive modeling.

### Key Components
Data Exploration (EDA): Comprehensive analysis of 253,680 survey records covering 22 health-related features like BMI, blood pressure, and physical activity.

Predictive Modeling: Comparison of multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) to find the most accurate classifier.

Web Application: A real-time dashboard built with Streamlit that allows users to input their health data and receive an instant risk assessment.

## 🛠️ Tech Stack
Data Processing: Pandas, NumPy

Machine Learning: Scikit-learn (Pipelines, Standard Scaler, Stratified K-Fold)

Analysis & Visualization: Matplotlib, Seaborn, SHAP

Deployment: Streamlit, Joblib

## 🚀 Getting Started
### 1. Prerequisites
Ensure you have Python installed, then install the required libraries:

pandas numpy scikit-learn matplotlib seaborn shap streamlit joblib

### 2. Dataset
The project uses the diabetes_012_health_indicators_BRFSS2015.csv dataset, which contains features like:

Health Conditions: High Blood Pressure, High Cholesterol, Heart Disease, Stroke.

Lifestyle Factors: BMI, Smoker status, Fruit/Veggie consumption, Physical Activity.

Demographics: Age, Education, Income, Sex.

### 3. Training the Model
Run the Jupyter Notebook Project 1 Diabetes Risk Predicter.ipynb to:

Load and clean the data.

Perform Exploratory Data Analysis.

Train and save the best-performing model to the models/ directory.

### 4. Running the App
Launch the Streamlit interface to interact with the model:

Bash
streamlit run diabetic_prediction_app.py

## 📊 Model Performance
The project utilizes evaluation metrics such as:

Accuracy, Precision, Recall, and F1-score to ensure reliable classifications across all three categories.

SHAP values to explain the impact of specific features (like BMI or Age) on the model's predictions.

## 🏥 Clinical Disclaimer
This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider.

### Powered by CDC Survey Records
