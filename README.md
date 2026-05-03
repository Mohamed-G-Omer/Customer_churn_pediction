# Customer Churn Prediction

A binary classification project that predicts which customers are likely to churn, enabling a marketing agency to proactively assign account managers to at-risk clients instead of assigning them randomly.

---

## Problem Statement

The agency has no system to identify which customers are at risk of leaving. This model uses historical customer data to predict churn probability, so account managers can be assigned where they are needed most.

---

## Dataset

**Training data:** `customer_churn.csv` — 900 customer records

| Feature | Description |
|---|---|
| Names | Name of the latest contact at the company |
| Age | Customer age |
| Total_Purchase | Total ad spend by the customer |
| Account_Manager | Whether an account manager is assigned (0 = No, 1 = Yes) |
| Years | Total years as a customer |
| Num_Sites | Number of websites using the service |
| Onboard_date | Date the contact was onboarded |
| Location | Client headquarters address |
| Company | Name of client company |
| Churn | Target variable — churned (1) or retained (0) |

**Prediction data:** `new_customers_1.csv` — 6 new customers without a churn label

---

## Methodology

### 1. Exploratory Data Analysis
- Churn distribution (bar chart + pie chart)
- Feature correlation heatmap (numeric features only)
- Missing value check
- Class balance check

### 2. Preprocessing
- Dropped non-numeric columns: `Names`, `Location`, `Company`, `Onboard_date`
- Train/test split: 80% training, 20% testing (`random_state=42`)
- Feature scaling with `StandardScaler` (mean=0, std=1)
- Added bias intercept column

### 3. Model — Logistic Regression (from scratch)
Implemented using gradient descent with:
- **Activation:** Sigmoid — `1 / (1 + e^-z)`
- **Loss:** Binary cross-entropy
- **Optimizer:** Gradient descent
- **Learning rate:** `α = 0.01`
- **Epochs:** 1000

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix heatmap
- Precision vs Recall curve
- Classification report

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~87–88% |
| Precision | ~88% |
| Recall | ~93% |
| F1-Score | ~90% |

The high recall ensures the model captures most customers who would have churned, minimizing missed interventions.

---

## Predictions on New Customers

| Customer | Churn Probability | Prediction | Action |
|---|---|---|---|
| Andrew Mccall | 24.10% | 0 | No Action Needed |
| Michele Wright | 92.81% | 1 | Assign Account Manager |
| Jeremy Chang | 77.77% | 1 | Assign Account Manager |
| Megan Ferguson | 87.93% | 1 | Assign Account Manager |
| Taylor Young | 34.63% | 0 | No Action Needed |
| Jessica Drake | 65.26% | 1 | Assign Account Manager |

**4 out of 6** new customers are at high risk and should be assigned an account manager.

---

## Project Structure

```
Customers Churn/
├── customerChurn.ipynb     # Main notebook — EDA, model, evaluation, predictions
├── customer_churn.csv      # Training dataset (900 records)
├── new_customers_1.csv     # New customers for prediction (6 records)
└── README.md
```

---

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

4. Open `customerChurn.ipynb` and run all cells (`Kernel > Restart & Run All`)

---
