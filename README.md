<h1 align="center">🔮 Customer Churn Prediction</h1>

<p align="center">
  <b>A from-scratch Logistic Regression model that predicts which customers are about to leave — so account managers can intervene before it's too late.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Seaborn-444876?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-88%25-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Recall-93%25-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/F1--Score-90%25-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Built%20from%20Scratch-red?style=flat-square" />
</p>

---

## The Problem

A marketing agency assigns account managers to clients **randomly** — there's no system to identify who's about to leave. By the time they notice, the customer is already gone.

**The question:** *Can we predict which customers are at risk of churning, so account managers go where they're actually needed?*

**The answer:** Yes — with **88% accuracy** and **93% recall**, catching nearly every at-risk customer before they walk out the door.

---

## What Makes This Different

This isn't a `model.fit()` one-liner. The **Logistic Regression is implemented from scratch** using:

```
Sigmoid Activation  →  Binary Cross-Entropy Loss  →  Gradient Descent Optimizer
         ↓                        ↓                            ↓
    σ(z) = 1/(1+e⁻ᶻ)      L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]    α = 0.01, 1000 epochs
```

Every weight update, every gradient calculation, every prediction — written from the ground up in NumPy. scikit-learn is used only for preprocessing and evaluation metrics.

---

## Dataset

**Training data:** `customer_churn.csv` — 900 customer records

| Feature | Description |
|:---|:---|
| `Age` | Customer age |
| `Total_Purchase` | Total ad spend by the customer |
| `Account_Manager` | Whether an account manager is currently assigned (0/1) |
| `Years` | Total years as a customer |
| `Num_Sites` | Number of websites using the service |
| `Churn` | **Target** — churned (1) or retained (0) |

Non-predictive columns (`Names`, `Location`, `Company`, `Onboard_date`) are dropped during preprocessing.

---

## Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│       EDA        │ ──▶ │  Preprocessing   │ ──▶ │    Training      │ ──▶ │   Evaluation     │
│                 │     │                 │     │                 │     │                 │
│ • Distribution  │     │ • Drop non-num  │     │ • Sigmoid       │     │ • Accuracy  88% │
│ • Correlation   │     │ • 80/20 split   │     │ • Cross-entropy │     │ • Recall    93% │
│ • Class balance │     │ • StandardScaler│     │ • Gradient desc │     │ • F1-Score  90% │
│ • Missing vals  │     │ • Bias column   │     │ • α=0.01, 1000ep│     │ • Confusion mtx │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Results

### Model Performance

| Metric | Score |
|:---|:---|
| **Accuracy** | ~88% |
| **Precision** | ~88% |
| **Recall** | **~93%** |
| **F1-Score** | ~90% |

> The high recall is the key metric here — it means the model catches **93% of customers who would have churned**. In a retention context, a missed churner costs far more than a false alarm.

### Visualisations generated in the notebook

- Churn distribution (bar + pie chart)
- Feature correlation heatmap
- Confusion matrix heatmap
- Precision vs Recall curve

---

## Predictions on New Customers

The trained model was applied to **6 unseen customers** with no churn label:

| Customer | Churn Probability | Prediction | Recommended Action |
|:---|:---:|:---:|:---|
| Andrew Mccall | 24.10% | ✅ Retain | No action needed |
| Michele Wright | **92.81%** | 🚨 Churn | **Assign account manager** |
| Jeremy Chang | **77.77%** | 🚨 Churn | **Assign account manager** |
| Megan Ferguson | **87.93%** | 🚨 Churn | **Assign account manager** |
| Taylor Young | 34.63% | ✅ Retain | No action needed |
| Jessica Drake | **65.26%** | 🚨 Churn | **Assign account manager** |

**Result:** 4 out of 6 new customers flagged as high-risk — actionable intelligence the agency didn't have before.

---

## Project Structure

```
Customer_churn_prediction/
├── customerChurn.ipynb     # Full pipeline — EDA, model, evaluation, predictions
├── customer_churn.csv      # Training dataset (900 records)
├── new_customers_1.csv     # New customers for prediction (6 records)
└── README.md
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/Mohamed-G-Omer/Customer_churn_pediction.git
cd Customer_churn_pediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# Launch
Jupyter Notebook customerChurn.ipynb
```

Then hit **Kernel → Restart & Run All** to reproduce every result.

---

## Key Takeaways

- **From-scratch implementation** deepens understanding of how logistic regression actually works under the hood — gradients, loss landscapes, convergence
- **Business framing matters** — the same model that outputs a probability becomes an actionable retention tool when framed as "assign an account manager: yes or no"
- **Recall over accuracy** — in churn problems, catching a churner matters more than overall accuracy. The 93% recall reflects that priority

---

**Mohamed Gamal Omer**
MSc Applied Data Science · HAN University of Applied Sciences

<p>
  <a href="https://www.linkedin.com/in/mohamed-omer-b57a84b6">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
  <a href="mailto:mohamed.geomer@gmail.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" />
  </a>
  <a href="https://github.com/Mohamed-G-Omer">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## License

This project is open source and available under the MIT License.
