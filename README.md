# Employee Attrition Predictor

A machine learning web application that predicts whether an employee is likely to leave a company, based on HR data. Built with Python, Flask, and deployed via IBM Watson Machine Learning.


## Overview

Employee attrition is the natural process by which employees leave the workforce — through resignation, retirement, or other reasons — without being immediately replaced. Early identification of at-risk employees allows HR teams to take proactive steps to improve retention.

This project trains and compares two classification models — **Logistic Regression** and **Random Forest Classifier** — on an HR dataset, evaluates them using 10-fold cross-validation, and serves predictions through a Flask web interface backed by IBM Watson ML.

**Authors:** Riddhi Gupta, Riya Gupta, Sonali Shripad Shanbhag


## Features

- Exploratory data analysis with seaborn/matplotlib visualizations
- Logistic Regression and Random Forest models, both validated with 10-fold cross-validation
- Feature importance analysis to identify key drivers of attrition
- Trained model serialized as a `.pkl` file
- Flask web app with a form-based UI for live predictions
- IBM Watson ML integration for cloud-based model deployment


## Dataset

**File:** `Employee_Attrition.csv`

The dataset contains HR records with the following features:

| Feature | Description |
|---|---|
| `satisfaction_level` | Employee satisfaction score (0–1) |
| `last_evaluation` | Most recent performance evaluation score (0–1) |
| `number_project` | Number of projects assigned |
| `average_montly_hours` | Average monthly hours worked |
| `time_spend_company` | Years spent at the company |
| `Work_accident` | Whether the employee had a workplace accident (0/1) |
| `promotion_last_5years` | Whether promoted in the last 5 years (0/1) |
| `Department` | Department (encoded as integer) |
| `salary` | Salary level (encoded as integer) |
| `left` | **Target variable** — whether the employee left (1) or stayed (0) |


## Models

### Logistic Regression
A baseline binary classification model trained on the preprocessed HR dataset, evaluated using 10-fold cross-validation.

### Random Forest Classifier
An ensemble tree-based model that generally achieves higher accuracy on tabular HR data. Feature importance scores are extracted and visualized to identify which factors most influence attrition.

Both models are compared on accuracy and cross-validation scores. The final model is saved to `employee_prediction.pkl`.


## Setup & Installation

### Prerequisites

- Python 3.7+
- An IBM Cloud account with Watson Machine Learning service enabled

### Install Dependencies

```bash
pip install flask numpy pandas scikit-learn requests pickle5
```

### IBM Watson ML Configuration

1. Create an IBM Cloud account and provision a Watson Machine Learning instance.
2. Deploy your trained model and obtain your **deployment URL** and **API key**.
3. In `app.py`, replace the placeholder values:

```python
API_KEY = "YOUR_IBM_CLOUD_API_KEY"
# And update the deployment URL in the requests.post() call
```

> ⚠️ **Security note:** Do not commit API keys to version control. Use environment variables instead.

### Run the App

```bash
python app.py
```

Then open your browser at `http://localhost:5000`.


## Usage

Fill out the prediction form with the following employee details:

- **Satisfaction Level** — a value between 0 and 1
- **Last Evaluation** — a value between 0 and 1
- **Number of Projects** — integer count
- **Average Monthly Hours** — integer
- **Time Spent with Company** — years (integer)
- **Work Accident** — Yes / No
- **Promotion in Last 5 Years** — Yes / No
- **Department** — select from dropdown
- **Salary Level** — Low / Medium / High

Click **Predict** to receive a result indicating whether the employee is likely to leave or stay.


## Testing the IBM Watson ML Endpoint Directly

Use `new.py` to send a test payload to the Watson ML API without the web UI:

```bash
python new.py
```

Edit the `values` array in the script to test different employee profiles.


## Results

Key findings from the analysis:

- **Satisfaction level**, **number of projects**, and **average monthly hours** are among the strongest predictors of attrition.
- Employees working very high or very low hours show elevated attrition risk.
- Lack of promotion over 5 years correlates with higher likelihood of leaving.

See `Report - Predicting Employee Attrition.pdf` and the visualizations (`hr_histogram_plots.png`, `random_forest.png`) for full details.


## License

This project was developed for academic purposes. See individual files for attribution.
