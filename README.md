# Customer Churn Analysis & Prediction

An end-to-end data science project on the public **Telco Customer Churn** dataset (7,043 customers, 21 columns). It covers data cleaning, exploratory analysis, statistics, feature engineering, model comparison and tuning, explainability with SHAP, business insights, and a Streamlit app that predicts churn probability for a single customer.

**Tech stack:** Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, Streamlit, Joblib

---

## 1. Problem

Customers who leave cost a telecom company recurring revenue, and winning a new customer is usually harder than keeping an existing one. The goal is to answer three questions:

1. Who is leaving, and what do those customers have in common?
2. Can we predict which customers are likely to leave before they do?
3. Which customers should a retention team contact first when it can only reach a few?

## 2. Dataset

The data is the IBM sample Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). Each row is a customer with demographics (gender, senior citizen, partner, dependents), account details (tenure, contract, payment method, paperless billing), services (phone, internet, security, backup, tech support, streaming) and charges. The target column is `Churn` (Yes/No). About **26.5%** of customers churned, so the classes are imbalanced.

## 3. Project workflow

| Stage | What was done |
|---|---|
| Cleaning | Checked shape, types, duplicates and category values; converted `TotalCharges` to numeric (11 blank values belong to customers with tenure 0); validated the target |
| Statistics | Mean, median, mode, variance, standard deviation, quartiles, IQR, skewness and kurtosis |
| EDA | 17 figures: distributions, churn rate per category, box, violin and KDE plots, scatter plot, multivariate heatmaps, correlation heatmap |
| Probability | Overall, conditional and joint probabilities, Bayes' theorem, normality check |
| Hypothesis tests | Welch t-test, Mann-Whitney U, chi-square, ANOVA, Pearson and Spearman correlation, Wilson confidence interval |
| Features | Tenure group, average monthly spend, total services, tech support and security flags, month-to-month flag, high charges flag, auto-payment flag |
| Models | Dummy baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost |
| Tuning | RandomizedSearchCV with stratified 5-fold cross-validation on ROC-AUC |
| Evaluation | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, false positive rate, Precision@10% and Lift@10% |
| Explainability | Permutation importance and SHAP (global and per-customer) |
| Deployment | Trained pipeline saved with Joblib and served through a Streamlit app |

## 4. Avoiding data leakage

The train/test split (80/20, stratified) happens **before** any fitted preprocessing. Imputation, scaling and one-hot encoding sit inside a Scikit-learn `Pipeline`, so they are learned from training folds only. The engineered features use only values from the same row, and the high-charges threshold is a fixed number rather than something learned from the data. The final model was selected by cross-validated ROC-AUC on the training set, not by test-set results. The test set was used once, for the final evaluation.

## 5. Key findings

These numbers come from running the project with `random_state=42`.

- **Churn rate:** 26.5% (95% confidence interval roughly 25.5% to 27.6%).
- **Contract type:** month-to-month customers churn at 42.7%, one-year at 11.3% and two-year at 2.8%. About 89% of all churners were on month-to-month contracts.
- **Tenure:** customers in their first 12 months churn at 47.4%, compared with 9.5% for those with more than 48 months.
- **Payment method:** electronic check users churn at 45.3%, compared with 15% to 17% for automatic bank transfer or credit card.
- **Services:** customers without tech support churn at 41.6%, compared with 15.2% for those with it. Online security shows a similar gap (41.8% vs 14.6%).
- **Internet service:** fiber optic customers churn at 41.9%.
- **Revenue:** churned customers account for 30.5% of total monthly revenue.

The chi-square tests show that contract, internet service, payment method and tech support are all strongly associated with churn. These are associations, not proof of cause.

## 6. Model results

All models were scored on the held-out test set (1,409 customers). The dummy baseline reaches 73.5% accuracy simply by predicting "no churn" for everyone, which is why accuracy alone is a poor metric here.

| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Dummy | 0.500 | 0.265 | 0.00 | 0.00 |
| Logistic Regression | 0.842 | 0.635 | 0.79 | 0.50 |
| Random Forest (tuned) | 0.843 | 0.644 | 0.79 | 0.53 |
| Gradient Boosting | 0.843 | 0.657 | 0.51 | 0.66 |
| **XGBoost (tuned)** | **0.848** | **0.664** | **0.80** | **0.52** |

The selected model is **tuned XGBoost**. At the default 0.5 threshold it catches about 80% of churners, at the cost of more false alarms (precision 0.52). The top 10% of customers by predicted risk are about **2.9 times** more likely to churn than average. Note that the strongest models are within about 0.01 ROC-AUC of each other, so the gain from tuning is real but small. A simple Logistic Regression is almost as good and easier to explain.

The threshold can be moved depending on cost. A higher threshold gives fewer, more accurate alerts. A lower one catches more churners but contacts more customers who would have stayed.

## 7. Explainability

SHAP and permutation importance agree on the main drivers: month-to-month contract, short tenure, fiber optic internet, electronic check payment, higher monthly charges, and no online security or tech support. SHAP explains what the model uses to predict, not what causes churn, so a business decision should not treat these as proven levers.

## 8. Business use

The project writes a priority list of the 10% of test customers with the highest predicted churn risk (`outputs/reports/retention_priority_customers.csv`). It also includes an illustrative ROI calculation, clearly labelled **hypothetical**, because the dataset has no offer costs or campaign results. Real retention ROI would need those numbers.

## 9. Streamlit app

The app takes tenure, contract, charges, internet service, payment method and the other customer attributes, and returns a churn probability.

```
python churn_simple.py
streamlit run app.py
```

## 10. Project structure

```
Customer_Churn_Data_Science/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_simple.py        # full pipeline, step by step
├── app.py                 # Streamlit app
├── requirements.txt
├── models/best_model.joblib
└── outputs/
    ├── figures/           # 17 charts
    ├── reports/           # statistics, tests, business insights
    └── metrics/           # model comparison
```

## 11. How to run

```
pip install -r requirements.txt
python churn_simple.py
python -m streamlit run app.py
```

The pipeline takes about two minutes and writes all figures, reports, metrics and the trained model.

## 12. Limitations and next steps

- The data is a single public snapshot, so there is no time dimension and no way to test how the model holds up on future customers.
- The class weights and the 0.5 threshold are defaults. A real deployment should set the threshold from the actual cost of an offer and the value of a retained customer.
- The model finds associations. Whether a retention offer works needs an A/B test.
- Next steps: probability calibration, threshold optimisation using cost, a FastAPI prediction endpoint, Docker packaging and experiment tracking with MLflow.

## Author

**Jyoti Ranjan Bhanja** | [LinkedIn](https://www.linkedin.com/in/jyotiranjanbhanja-ds) | [GitHub](https://github.com/JyotiAnalytics)

