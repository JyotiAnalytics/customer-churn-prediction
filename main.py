"""
Customer Churn Analysis & Prediction - Complete Data Science Project (Telco dataset)

Run:   python main.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
Needs: see requirements.txt

Stages covered: setup, loading, inspection, cleaning, missing values, duplicates,
descriptive stats, EDA, visualisation, probability, hypothesis tests, correlation,
feature engineering, encoding/scaling, train-test split, ML models, tuning,
evaluation, explainability (SHAP + permutation importance), business analysis,
model saving for deployment (see app.py).

Leakage rule: the split happens BEFORE any fitted preprocessing. Imputation, scaling and
encoding live inside a scikit-learn Pipeline, so they are learned from training folds only.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # optional
    XGBClassifier = None
try:
    import shap
except ImportError:  # optional
    shap = None

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
TARGET = "Churn"

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
AUTO_PAY = ["Bank transfer (automatic)", "Credit card (automatic)"]
HIGH_CHARGE_THRESHOLD = 70.0  # fixed business rule, so nothing is learned from the data


# ----------------------------------------------------------------------------
# Stage 1: project setup
# ----------------------------------------------------------------------------
def setup_dirs(base: Path):
    dirs = {k: base / "outputs" / k for k in ("figures", "reports", "metrics")}
    dirs["models"] = base / "models"
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Stages 2-6: loading, inspection, cleaning, missing values, duplicates
# ----------------------------------------------------------------------------
def load_data(path):
    return pd.read_csv(path)


def inspect(df, out: Path):
    lines = []
    lines.append(f"Shape: {df.shape}")
    lines.append(f"Columns: {list(df.columns)}")
    lines.append("\nDtypes:\n" + df.dtypes.to_string())
    lines.append("\nMissing values:\n" + df.isnull().sum().to_string())
    lines.append(f"\nDuplicate rows: {df.duplicated().sum()}")
    lines.append(f"Duplicate customerIDs: {df['customerID'].duplicated().sum()}")
    lines.append("\nTarget distribution:\n" + df[TARGET].value_counts().to_string())
    # inconsistent categorical values
    for c in df.select_dtypes("object").columns.difference(["customerID", "TotalCharges"]):
        lines.append(f"\n{c}: {sorted(df[c].dropna().unique().tolist())}")
    text = "\n".join(lines)
    (out / "data_inspection.txt").write_text(text)
    print(text)


def clean(df):
    df = df.copy()
    df = df.drop_duplicates()
    # TotalCharges contains blank strings (new customers with tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    print("\nTotalCharges NaN after conversion:", df["TotalCharges"].isnull().sum())
    print("Those rows have tenure==0:", (df.loc[df["TotalCharges"].isnull(), "tenure"] == 0).all())
    # Strip whitespace in text columns
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    # Validate target
    assert set(df[TARGET].unique()) <= {"Yes", "No"}, "Unexpected target values"
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # NOTE: missing TotalCharges are NOT filled here. A median imputer inside the pipeline
    # learns the fill value from training data only.
    return df


# ----------------------------------------------------------------------------
# Stage 7: descriptive statistics
# ----------------------------------------------------------------------------
def descriptive_stats(df, out: Path):
    rows = {}
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        s = df[c].dropna()
        q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
        rows[c] = {
            "mean": s.mean(), "median": s.median(), "mode": s.mode().iloc[0],
            "variance": s.var(), "std": s.std(), "min": s.min(), "max": s.max(),
            "Q1": q1, "Q2": q2, "Q3": q3, "IQR": q3 - q1,
            "skewness": s.skew(), "kurtosis": s.kurt(),
        }
    res = pd.DataFrame(rows).T.round(3)
    res.to_csv(out / "descriptive_statistics.csv")
    print("\nDescriptive statistics:\n", res)
    return res


# ----------------------------------------------------------------------------
# Stages 8-9, 12: EDA, visualisation, correlation
# ----------------------------------------------------------------------------
def eda(df, figdir: Path):
    d = df.copy()
    d["ChurnFlag"] = (d[TARGET] == "Yes").astype(int)

    # Univariate
    plt.figure(figsize=(5, 4))
    sns.countplot(data=d, x=TARGET)
    plt.title("Churn distribution")
    savefig(figdir / "01_churn_distribution.png")

    cats = ["gender", "Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), cats):
        sns.countplot(data=d, x=c, ax=ax)
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(f"{c} distribution")
    savefig(figdir / "02_categorical_distributions.png")

    nums = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, nums):
        sns.histplot(d[c].dropna(), kde=True, ax=ax)
        ax.set_title(f"{c} histogram")
    savefig(figdir / "03_numeric_histograms.png")

    # Bivariate: churn rate (%) per category
    bi = ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract", "InternetService",
          "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    for ax, c in zip(axes.ravel(), bi):
        rate = d.groupby(c)["ChurnFlag"].mean().mul(100).sort_values()
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_ylabel("Churn %")
        ax.set_title(f"Churn rate by {c}")
        ax.tick_params(axis="x", rotation=15)
    savefig(figdir / "04_churn_rate_by_category.png")

    # Stacked bar (contract x churn)
    ct = pd.crosstab(d["Contract"], d[TARGET], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, figsize=(6, 4))
    plt.ylabel("%")
    plt.title("Churn share by contract (stacked)")
    savefig(figdir / "05_stacked_contract.png")

    # Boxplot, violin, KDE
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.boxplot(data=d, x=TARGET, y="MonthlyCharges", ax=axes[0])
    sns.violinplot(data=d, x=TARGET, y="tenure", ax=axes[1])
    sns.kdeplot(data=d, x="MonthlyCharges", hue=TARGET, fill=True, common_norm=False, ax=axes[2])
    axes[0].set_title("MonthlyCharges by churn (box)")
    axes[1].set_title("Tenure by churn (violin)")
    axes[2].set_title("MonthlyCharges KDE by churn")
    savefig(figdir / "06_box_violin_kde.png")

    # Scatter
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=d, x="tenure", y="MonthlyCharges", hue=TARGET, alpha=0.5)
    plt.title("Tenure vs MonthlyCharges")
    savefig(figdir / "07_scatter_tenure_monthly.png")

    # Multivariate
    piv = d.pivot_table(index="Contract", columns="InternetService", values="ChurnFlag", aggfunc="mean") * 100
    plt.figure(figsize=(6, 4))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Churn % : Contract x InternetService")
    savefig(figdir / "08_multivariate_contract_internet.png")

    piv2 = d.pivot_table(index="PaymentMethod", columns="Contract", values="ChurnFlag", aggfunc="mean") * 100
    plt.figure(figsize=(8, 4))
    sns.heatmap(piv2, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Churn % : PaymentMethod x Contract")
    savefig(figdir / "09_multivariate_payment_contract.png")

    piv3 = d.pivot_table(index="TechSupport", columns="Contract", values="ChurnFlag", aggfunc="mean") * 100
    plt.figure(figsize=(7, 4))
    sns.heatmap(piv3, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Churn % : TechSupport x Contract")
    savefig(figdir / "10_multivariate_techsupport_contract.png")

    d["TenureBand"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    piv4 = d.pivot_table(index="Contract", columns="TenureBand", values="ChurnFlag", aggfunc="mean", observed=False) * 100
    plt.figure(figsize=(7, 4))
    sns.heatmap(piv4, annot=True, fmt=".1f", cmap="Reds")
    plt.title("Churn % : Contract x Tenure band")
    savefig(figdir / "11_multivariate_contract_tenure.png")

    # Correlation (numeric + target flag)
    corr = d[["tenure", "MonthlyCharges", "TotalCharges", "ChurnFlag"]].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    savefig(figdir / "12_correlation_heatmap.png")


# ----------------------------------------------------------------------------
# Stage 10: probability analysis
# ----------------------------------------------------------------------------
def probability_analysis(df, out: Path):
    churn = df[TARGET] == "Yes"
    res = {}
    res["P(Churn)"] = churn.mean()
    for label, mask in {
        "Month-to-month": df["Contract"] == "Month-to-month",
        "Fiber optic": df["InternetService"] == "Fiber optic",
        "No tech support": df["TechSupport"] == "No",
        "Two year": df["Contract"] == "Two year",
    }.items():
        res[f"P(Churn | {label})"] = churn[mask].mean()
        res[f"P(Churn AND {label})"] = (churn & mask).mean()  # joint probability

    # Bayes: P(M2M | Churn) = P(Churn | M2M) * P(M2M) / P(Churn)
    m2m = df["Contract"] == "Month-to-month"
    bayes = churn[m2m].mean() * m2m.mean() / churn.mean()
    res["Bayes P(Month-to-month | Churn)"] = bayes
    res["Direct check P(Month-to-month | Churn)"] = m2m[churn].mean()

    # Distribution: is tenure normal? (Shapiro on a sample, skew/kurtosis)
    sample = df["tenure"].sample(min(5000, len(df)), random_state=RANDOM_STATE)
    res["Shapiro p-value (tenure)"] = stats.shapiro(sample).pvalue
    out_df = pd.Series(res).round(4)
    out_df.to_csv(out / "probability_analysis.csv", header=["value"])
    print("\nProbability analysis:\n", out_df)
    return out_df


# ----------------------------------------------------------------------------
# Stage 11: hypothesis testing
# ----------------------------------------------------------------------------
def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def hypothesis_tests(df, out: Path):
    rows = []
    yes = df[df[TARGET] == "Yes"]
    no = df[df[TARGET] == "No"]

    t, p = stats.ttest_ind(yes["MonthlyCharges"], no["MonthlyCharges"], equal_var=False)
    rows.append(("Welch t-test", "MonthlyCharges: churn vs non-churn", t, p))

    u, p = stats.mannwhitneyu(yes["MonthlyCharges"], no["MonthlyCharges"], alternative="two-sided")
    rows.append(("Mann-Whitney U", "MonthlyCharges: churn vs non-churn", u, p))

    for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "SeniorCitizen"]:
        chi2, p, dof, _ = chi2_contingency_wrapper(pd.crosstab(df[c], df[TARGET]))
        rows.append(("Chi-square", f"{c} vs Churn (dof={dof})", chi2, p))

    groups = [g["MonthlyCharges"].values for _, g in df.groupby("Contract")]
    f, p = stats.f_oneway(*groups)
    rows.append(("ANOVA", "MonthlyCharges across Contract types", f, p))

    d = df.dropna(subset=["TotalCharges"])
    r, p = stats.pearsonr(d["tenure"], d["TotalCharges"])
    rows.append(("Pearson correlation", "tenure vs TotalCharges", r, p))
    rho, p = stats.spearmanr(df["tenure"], df["MonthlyCharges"])
    rows.append(("Spearman correlation", "tenure vs MonthlyCharges", rho, p))

    res = pd.DataFrame(rows, columns=["test", "question", "statistic", "p_value"])
    res["significant_at_0.05"] = res["p_value"] < 0.05
    res.to_csv(out / "hypothesis_tests.csv", index=False)
    print("\nHypothesis tests (association, not causation):\n", res.to_string(index=False))

    lo, hi = wilson_ci((df[TARGET] == "Yes").sum(), len(df))
    print(f"95% Wilson CI for overall churn rate: [{lo:.4f}, {hi:.4f}]")
    return res


def chi2_contingency_wrapper(table):
    chi2, p, dof, expected = stats.chi2_contingency(table)
    return chi2, p, dof, expected


# ----------------------------------------------------------------------------
# Stage 13: feature engineering (row-level only, nothing learned from data)
# ----------------------------------------------------------------------------
def add_features(df):
    d = df.copy()
    d["TenureGroup"] = pd.cut(d["tenure"], bins=[-1, 12, 24, 48, 72],
                              labels=["New", "Short", "Medium", "Long"]).astype(str)
    # Average spend per month of tenure; tenure 0 -> fall back to MonthlyCharges
    safe_tenure = d["tenure"].replace(0, np.nan)
    d["AvgMonthlySpend"] = (d["TotalCharges"] / safe_tenure).fillna(d["MonthlyCharges"])
    d["TotalServices"] = sum((d[c] == "Yes").astype(int) for c in SERVICE_COLS)
    d["TotalServices"] += (d["InternetService"] != "No").astype(int)
    d["HasTechSupport"] = (d["TechSupport"] == "Yes").astype(int)
    d["HasOnlineSecurity"] = (d["OnlineSecurity"] == "Yes").astype(int)
    d["IsMonthToMonth"] = (d["Contract"] == "Month-to-month").astype(int)
    d["HighMonthlyCharges"] = (d["MonthlyCharges"] > HIGH_CHARGE_THRESHOLD).astype(int)
    d["AutoPayment"] = d["PaymentMethod"].isin(AUTO_PAY).astype(int)
    return d


# ----------------------------------------------------------------------------
# Stage 14-16: preprocessing and models
# ----------------------------------------------------------------------------
def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def get_models(pos_weight):
    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE)
    return models


def tuning_spaces():
    spaces = {
        "RandomForest": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [3, 5, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    }
    if XGBClassifier is not None:
        spaces["XGBoost"] = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [2, 3, 4, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        }
    return spaces


# ----------------------------------------------------------------------------
# Stage 18: evaluation
# ----------------------------------------------------------------------------
def precision_lift_at_k(y_true, proba, k=0.10):
    n = max(1, int(len(y_true) * k))
    idx = np.argsort(-proba)[:n]
    prec = np.asarray(y_true)[idx].mean()
    return prec, prec / np.mean(y_true)


def evaluate(name, pipe, X_te, y_te, threshold=0.5):
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    p10, l10 = precision_lift_at_k(y_te, proba, 0.10)
    return {
        "model": name,
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "log_loss": log_loss(y_te, np.clip(proba, 1e-6, 1 - 1e-6)),
        "false_positive_rate": fp / (fp + tn),
        "precision@10%": p10,
        "lift@10%": l10,
    }, proba, pred


def plot_curves(results_proba, y_te, figdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, proba in results_proba.items():
        fpr, tpr, _ = roc_curve(y_te, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, proba):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, proba)
        axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(title="ROC curves", xlabel="FPR", ylabel="TPR")
    axes[1].set(title="Precision-Recall curves", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    savefig(figdir / "13_roc_pr_curves.png")


# ----------------------------------------------------------------------------
# Stage 19: explainability
# ----------------------------------------------------------------------------
def explain(best_name, best_pipe, X_te, y_te, figdir: Path, reportdir: Path):
    # Permutation importance on held-out data (works for any model)
    perm = permutation_importance(best_pipe, X_te, y_te, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    pi = pd.Series(perm.importances_mean, index=X_te.columns).sort_values(ascending=False)
    pi.to_csv(reportdir / "permutation_importance.csv", header=["roc_auc_drop"])
    plt.figure(figsize=(8, 6))
    pi.head(15).iloc[::-1].plot(kind="barh")
    plt.title(f"Permutation importance (test, ROC-AUC drop) - {best_name}")
    savefig(figdir / "14_permutation_importance.png")

    if shap is None:
        print("shap not installed - skipping SHAP")
        return
    pre, model = best_pipe.named_steps["pre"], best_pipe.named_steps["model"]
    Xt = pre.transform(X_te)
    names = pre.get_feature_names_out()
    Xt_df = pd.DataFrame(Xt, columns=names)
    sample = Xt_df.sample(min(500, len(Xt_df)), random_state=RANDOM_STATE)
    try:
        if best_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, Xt_df)
            sv = explainer.shap_values(sample)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(sample)
        if isinstance(sv, list):          # older shap: list per class
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:  # newer shap: (n, features, classes)
            sv = sv[:, :, 1]
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=15)
        savefig(figdir / "15_shap_summary_beeswarm.png")
        plt.figure()
        shap.summary_plot(sv, sample, plot_type="bar", show=False, max_display=15)
        savefig(figdir / "16_shap_global_importance.png")
        # Individual explanation: highest-risk customer in the sample
        i = int(np.argmax(model.predict_proba(sample)[:, 1]))
        contrib = pd.Series(sv[i], index=names).sort_values(key=abs, ascending=False).head(10)
        contrib.to_csv(reportdir / "shap_top_customer_explanation.csv", header=["shap_value"])
        print("\nTop SHAP drivers for the highest-risk sampled customer:\n", contrib.round(3))
    except Exception as e:  # SHAP version quirks should not kill the pipeline
        print("SHAP skipped:", e)


# ----------------------------------------------------------------------------
# Stage 13b: business analysis
# ----------------------------------------------------------------------------
def business_analysis(df, X_te, proba, reportdir: Path):
    lines = []
    d = df.copy()
    d["ChurnFlag"] = (d[TARGET] == "Yes").astype(int)
    lines.append(f"Churn rate: {d['ChurnFlag'].mean():.1%}")
    by_contract = d.groupby("Contract")["ChurnFlag"].mean().sort_values(ascending=False)
    lines.append("Churn by contract:\n" + by_contract.apply(lambda x: f"{x:.1%}").to_string())
    by_pay = d.groupby("PaymentMethod")["ChurnFlag"].mean().sort_values(ascending=False)
    lines.append("Churn by payment method:\n" + by_pay.apply(lambda x: f"{x:.1%}").to_string())
    by_tenure = d.groupby(pd.cut(d["tenure"], [-1, 12, 24, 48, 72]), observed=False)["ChurnFlag"].mean()
    lines.append("Churn by tenure band:\n" + by_tenure.apply(lambda x: f"{x:.1%}").to_string())
    for c in ["TechSupport", "OnlineSecurity"]:
        lines.append(f"Churn by {c}:\n" + d.groupby(c)["ChurnFlag"].mean().apply(lambda x: f"{x:.1%}").to_string())

    churned = d[d["ChurnFlag"] == 1]
    lines.append(f"\nMonthly revenue associated with churned customers: {churned['MonthlyCharges'].sum():,.0f} "
                 f"({churned['MonthlyCharges'].sum() / d['MonthlyCharges'].sum():.1%} of total monthly revenue)")

    # Priority list: top 10% by predicted churn risk on the held-out set
    t = X_te.copy()
    t["churn_probability"] = proba
    k = max(1, int(len(t) * 0.10))
    top = t.sort_values("churn_probability", ascending=False).head(k)
    top.to_csv(reportdir / "retention_priority_customers.csv")
    lines.append(f"\nRetention priority: top {k} test customers by predicted risk "
                 f"(avg MonthlyCharges {top['MonthlyCharges'].mean():.1f}).")

    # HYPOTHETICAL scenario - the dataset has no campaign cost/outcome data
    cost, save_rate, months_kept = 10.0, 0.20, 12
    true_churners = (d.loc[top.index, "ChurnFlag"]).sum() if set(top.index) <= set(d.index) else np.nan
    if not np.isnan(true_churners):
        saved_value = true_churners * save_rate * top["MonthlyCharges"].mean() * months_kept
        campaign_cost = k * cost
        lines.append(
            "\n[HYPOTHETICAL - assumptions, not data] offer cost 10/customer, 20% of targeted churners "
            f"retained for 12 months: value {saved_value:,.0f} vs cost {campaign_cost:,.0f}."
        )
    text = "\n".join(lines)
    (reportdir / "business_insights.txt").write_text(text)
    print("\n" + text)


# ----------------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------------
def main(data_path, base_dir="."):
    base = Path(base_dir)
    dirs = setup_dirs(base)

    raw = load_data(data_path)
    inspect(raw, dirs["reports"])
    df = clean(raw)
    descriptive_stats(df, dirs["reports"])
    eda(df, dirs["figures"])
    probability_analysis(df, dirs["reports"])
    hypothesis_tests(df, dirs["reports"])

    feat = add_features(df)
    y = (feat[TARGET] == "Yes").astype(int)
    X = feat.drop(columns=[TARGET, "customerID"])

    # Stage 15: split BEFORE any fitting
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain {X_tr.shape}, Test {X_te.shape}, churn rate train={y_tr.mean():.3f} test={y_te.mean():.3f}")

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Stage 16: baseline comparison with CV on the training set
    pipes, cv_scores = {}, {}
    for name, model in get_models(pos_weight).items():
        pipe = Pipeline([("pre", build_preprocessor(X_tr)), ("model", model)])
        cv_scores[name] = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        pipes[name] = pipe
        print(f"CV ROC-AUC {name:20s}: {cv_scores[name]:.4f}")

    # Stage 17: hyperparameter tuning (randomized search, stratified CV)
    for name, space in tuning_spaces().items():
        search = RandomizedSearchCV(pipes[name], space, n_iter=15, scoring="roc_auc",
                                    cv=cv, n_jobs=-1, random_state=RANDOM_STATE)
        search.fit(X_tr, y_tr)
        pipes[f"{name}_Tuned"] = search.best_estimator_
        cv_scores[f"{name}_Tuned"] = search.best_score_
        print(f"Tuned {name}: CV ROC-AUC {search.best_score_:.4f}  params={search.best_params_}")

    # Final fit of untuned pipelines, then evaluation on the untouched test set
    results, probas = [], {}
    for name, pipe in pipes.items():
        if not name.endswith("_Tuned"):
            pipe.fit(X_tr, y_tr)
        metrics, proba, pred = evaluate(name, pipe, X_te, y_te)
        metrics["cv_roc_auc"] = cv_scores[name]
        results.append(metrics)
        probas[name] = proba
    res_df = pd.DataFrame(results).set_index("model").round(4)
    res_df.to_csv(dirs["metrics"] / "model_comparison.csv")
    print("\nTest-set comparison:\n", res_df.to_string())

    # Model selection uses CV score (not the test set); Dummy excluded
    candidates = {k: v for k, v in cv_scores.items() if k != "Dummy"}
    best_name = max(candidates, key=candidates.get)
    best_pipe = pipes[best_name]
    print(f"\nSelected model (best CV ROC-AUC): {best_name}")

    plot_curves({k: v for k, v in probas.items() if k in ("Dummy", "LogisticRegression", best_name)},
                y_te, dirs["figures"])

    best_pred = (probas[best_name] >= 0.5).astype(int)
    report = classification_report(y_te, best_pred, target_names=["No churn", "Churn"])
    (dirs["reports"] / "classification_report.txt").write_text(report)
    print(report)
    cm = confusion_matrix(y_te, best_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix - {best_name}")
    savefig(dirs["figures"] / "17_confusion_matrix.png")

    explain(best_name, best_pipe, X_te, y_te, dirs["figures"], dirs["reports"])
    business_analysis(feat, X_te, probas[best_name], dirs["reports"])

    # Stage 20: save the model for deployment
    joblib.dump(best_pipe, dirs["models"] / "best_model.joblib")
    meta = {"best_model": best_name, "feature_columns": X.columns.tolist(),
            "test_metrics": res_df.loc[best_name].to_dict()}
    (dirs["metrics"] / "best_model_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nSaved {dirs['models'] / 'best_model.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    main(args.data, args.out)
