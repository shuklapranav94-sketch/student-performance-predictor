# 🎓 Student Performance Predictor

A Machine Learning based web application that predicts a student's academic performance using academic, behavioural, learning, and engagement-related information.

The project uses Machine Learning models for classification and provides prediction results, confidence scores, performance probabilities, personalized recommendations, model evaluation, and model comparison through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Student academic performance can be influenced by several factors such as study hours, attendance, assignment completion, motivation, learning style, internet access, online courses, discussion participation, and stress level.

The objective of this project is to develop an end-to-end Machine Learning system that can analyze these factors and predict the student's expected performance level.

The application provides an easy-to-use interface where users can enter student information and receive:

- 🎯 Predicted Performance
- 📊 Prediction Confidence
- 📈 Performance Probability
- 💡 Personalized Recommendations
- 📚 Student Academic Analysis
- 🤖 Model Performance Metrics
- 📊 Confusion Matrix
- 🏆 Machine Learning Model Comparison
- 📥 Downloadable Student Report

---

# 🎯 Problem Statement

Educational institutions collect a large amount of student-related data, but identifying students who may require additional academic support can be challenging.

This project aims to use Machine Learning to analyze different student-related features and predict their academic performance.

The system can provide an early indication of performance and generate recommendations based on the selected student inputs.

---

# 🚀 Project Objectives

The main objectives of this project are:

1. Collect and preprocess student-related data.
2. Perform Exploratory Data Analysis (EDA).
3. Perform feature engineering.
4. Train Machine Learning classification models.
5. Evaluate different Machine Learning algorithms.
6. Select the best-performing model.
7. Save the trained model.
8. Build an interactive Streamlit web application.
9. Predict student performance.
10. Display prediction probability and confidence.
11. Generate personalized recommendations.
12. Provide model evaluation and comparison dashboards.
13. Generate a downloadable student performance report.

---

# ✨ Features

## 🤖 Student Performance Prediction

The application accepts student-related information such as:

- Study Hours
- Attendance
- Assignment Completion
- Age
- Gender
- Learning Style
- Resources
- Internet Access
- Online Courses
- Motivation
- Discussion Participation
- Extracurricular Activities
- EduTech Usage
- Stress Level

The trained Machine Learning model uses these features to predict the student's performance category.

---

## 🎯 Prediction Confidence

The application displays the confidence/probability associated with the predicted performance class when the selected model supports probability prediction.

---

## 📊 Performance Probability

The application displays the probability distribution for the available performance classes.

Example performance categories:

- Low Performance
- Below Average Performance
- Good Performance
- Excellent Performance

---

## 💡 Personalized Recommendations

Based on the student's inputs, the application provides recommendations related to:

- Study hours
- Attendance
- Assignment completion
- Online courses
- Motivation
- Discussion participation
- Extracurricular activities
- Stress level
- Internet access

---

## 📈 Student Analysis Dashboard

After prediction, the application displays:

- Student profile
- Academic indicators
- Learning activity
- Engagement summary
- Performance probability chart

---

## 📊 Model Performance Dashboard

The application provides model evaluation metrics including:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🏆 Model Comparison

Different Machine Learning models can be compared using:

- Accuracy
- Precision
- Recall
- F1 Score

The application identifies the model with the highest F1 Score as the best-performing model.

---

## 📥 Download Student Report

Users can download a CSV report containing:

- Student information
- Academic information
- Learning information
- Engagement information
- Predicted performance
- Prediction confidence
- Personalized recommendations

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| Joblib | Model Saving and Loading |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |
| Jupyter Notebook | Data Analysis and Experimentation |
| Git & GitHub | Version Control |

---

# 🔄 Machine Learning Workflow

The complete workflow of the project is:

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Streamlit Application
        ↓
Student Input
        ↓
Prediction
        ↓
Analysis & Recommendations


#PROJECT STRUCTURE 

student-performance-predictor/
│
├── app/
│   └── app.py
│
├── data/
│   └── student_performance.csv
│
├── models/
│   ├── model_metrics.pkl
│   ├── confusion_matrix.pkl
│   └── model_comparison_results.csv
│
├── notebooks/
│   └── student_performance_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── .gitignore
├── README.md
└── requirements.txt