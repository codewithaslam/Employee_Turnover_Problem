# Employee Turnover Prediction using Logistic Regression

## Overview

This project focuses on predicting employee turnover (attrition) using Machine Learning techniques. Employee turnover prediction helps organizations identify employees who are likely to leave the company, enabling proactive retention strategies.

The project implements and compares three Logistic Regression models:

* Baseline Logistic Regression
* L1 Regularized Logistic Regression (Lasso)
* L2 Regularized Logistic Regression (Ridge)

The models are evaluated using Accuracy, Precision, Recall, and F1-Score to determine the best-performing approach.

---

## Features

* Employee turnover prediction
* Data preprocessing and train-test splitting
* Logistic Regression implementation
* L1 (Lasso) Regularization
* L2 (Ridge) Regularization
* Model comparison and evaluation
* Performance reporting using classification metrics

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* NumPy
* Logistic Regression
* Classification Metrics

---

## Dataset

The project uses an employee turnover dataset containing various employee-related attributes and a target column:

* Employee_Turnover (Target Variable)

The target variable indicates whether an employee stays with or leaves the organization.

---

## Workflow

1. Load employee turnover dataset.
2. Separate features and target variable.
3. Split the dataset into training and testing sets.
4. Train three Logistic Regression models:

   * Baseline Logistic Regression
   * Lasso Regression (L1 Regularization)
   * Ridge Regression (L2 Regularization)
5. Generate predictions on the test dataset.
6. Evaluate models using:

   * Accuracy
   * Precision
   * Recall
   * F1-Score
7. Identify the best-performing model.

---

## Model Evaluation Metrics

The models are evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

---

## Results

| Model                        | Accuracy |
| ---------------------------- | -------- |
| Baseline Logistic Regression | 89.63%   |
| Lasso Regression (L1)        | 89.63%   |
| Ridge Regression (L2)        | 89.63%   |

### Best Model

**Baseline Logistic Regression**

**Accuracy:** 89.63%

---

## Project Structure

```
EmployeeTurnoverPrediction/
│
├── employee_turnover.csv
├── employee_turnover.py
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd EmployeeTurnoverPrediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python employee_turnover.py
```

---

## Future Enhancements

* Hyperparameter tuning
* Cross-validation
* Feature selection
* Streamlit web application
* Model deployment
* Advanced classification algorithms (Random Forest, XGBoost, SVM)

---

## Author

**Muhammad Aslam Shanavas**

B.Tech Computer Science and Engineering
Albertian Institute of Science and Technology (AISAT)

---

## License

This project is developed for educational and learning purposes.
