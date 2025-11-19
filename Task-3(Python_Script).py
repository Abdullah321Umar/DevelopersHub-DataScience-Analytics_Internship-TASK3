"""
   Task 3:
   — Customer Churn Prediction (Bank Customers) 

"""



import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin



# ------------------------- Path -------------------------------

default_windows_path = r"C:/Users/Abdullah Umer/Desktop/DevelopersHub Corporation Internship/TASK 3/Churn_Modelling DataSet.csv"


# Fallback (when running in a different environment)
fallback_path = "/mnt/data/Churn_Modelling DataSet.csv"

csv_path = default_windows_path if os.path.exists(default_windows_path) else fallback_path
if not os.path.exists(csv_path):
    print("CSV not found at default paths. Please edit csv_path variable to point to your CSV file.")
    sys.exit(1)


# Directory to save outputs (same directory as CSV)
base_dir = os.path.dirname(csv_path)
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)


# ---------- Load dataset ----------
df = pd.read_csv(csv_path)
print(f"Loaded dataset from {csv_path} — shape: {df.shape}")


# ---------- Quick cleaning ----------
# Remove ID-like columns that aren't helpful to ML
drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
df_clean = df.drop(columns=drop_cols).copy()


# Fill missing values if any (median imputation for numerics)
if df_clean.isnull().any().any():
    print("Missing values detected — applying median imputation for numerical columns.")
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

# Create a few derived features (useful for interpretation)
df_clean["BalanceZero"] = (df_clean["Balance"] == 0).astype(int)
df_clean["HighBalance"] = (df_clean["Balance"] > df_clean["Balance"].median()).astype(int)
df_clean["Senior"] = (df_clean["Age"] >= 60).astype(int)




# ---------- Split features & target ----------
TARGET = "Exited"
if TARGET not in df_clean.columns:
    raise ValueError(f"Expected target column '{TARGET}' in dataset.")

X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]




# ---------- LabelEncoder wrapper for ColumnTransformer ----------
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()
    def fit(self, X, y=None):
        arr = np.asarray(X).ravel()
        self.le.fit(arr)
        return self
    def transform(self, X):
        arr = np.asarray(X).ravel()
        return self.le.transform(arr).reshape(-1, 1)
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ---------- Column transformer ----------
cat_cols = [c for c in ["Geography", "Gender"] if c in X.columns]
preprocessor = ColumnTransformer([
    ("onehot_geo", OneHotEncoder(drop="first", sparse_output=False), ["Geography"]) if "Geography" in X.columns else ("noop", "passthrough", []),
    ("label_gender", LabelEncoderWrapper(), ["Gender"]) if "Gender" in X.columns else ("noop2", "passthrough", [])
], remainder="passthrough")



# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
preprocessor.fit(X_train)  # fit on training data

# Helper to derive post-transform feature names
def get_feature_names_from_preprocessor(preprocessor, X):
    feature_names = []
    # preprocessor.transformers_ available after fit
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            if trans == "passthrough":
                remainder_cols = [c for c in X.columns if c not in sum([list(t[2]) for t in preprocessor.transformers_ if t[0]!="remainder"], [])]
                feature_names.extend(remainder_cols)
            else:
                feature_names.extend(cols)
        else:
            if hasattr(trans, "get_feature_names_out"):
                try:
                    out = trans.get_feature_names_out(cols)
                    feature_names.extend(out.tolist())
                except Exception:
                    try:
                        out = trans.get_feature_names(cols)
                        feature_names.extend(out)
                    except Exception:
                        feature_names.extend(cols)
            else:
                feature_names.extend(cols)
    return feature_names

feature_names = get_feature_names_from_preprocessor(preprocessor, X_train)


# ---------- Scale data ----------
scaler = StandardScaler()
X_train_trans = preprocessor.transform(X_train)
scaler.fit(X_train_trans)
X_test_trans = preprocessor.transform(X_test)
X_test_scaled = scaler.transform(X_test_trans)



# ---------- Models ----------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1200, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

pipelines = {}
for name, model in models.items():
    # build a simple pipeline wrapper so we can call predict_proba on raw X
    pipelines[name] = Pipeline([("pre", preprocessor), ("scale", scaler), ("model", model)])
    pipelines[name].fit(X_train, y_train)


# ---------- Evaluate models ----------
results = {}
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:,1]
    except Exception:
        # fallback
        y_proba = pipe.predict(X_test)
    roc = roc_auc_score(y_test, y_proba)
    results[name] = {"roc_auc": roc, "report": classification_report(y_test, y_pred, output_dict=True)}

best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
best_pipeline = pipelines[best_model_name]
best_model = best_pipeline.named_steps["model"]
print("\nModel ROC-AUC scores:")
for name, res in results.items():
    print(f" - {name}: ROC-AUC = {res['roc_auc']:.4f}")
print(f">>> Best model selected: {best_model_name}")

# ---------- Permutation importance on the scaled test matrix ----------
perm = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=15, random_state=42, n_jobs=-1)
perm_idx = perm.importances_mean.argsort()[::-1]





# ---------- Visualization settings ----------
plt.style.use("dark_background")  # black background
# friendly dark palette (cycled)
palette = [
    "#01b3a7", "#ff7b00", "#ffd60a", "#9b5de5", "#f15bb5",
    "#00bbf9", "#00f5d4", "#ff6b6b", "#8ac926", "#1982c4"
]

def savefig(fig, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")


# --------------------- 20 distinct visualizations  ---------------------

# 1) Churn count bar
fig = plt.figure(figsize=(6,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
counts = df_clean["Exited"].value_counts().sort_index()
ax.bar(["Stayed","Exited"], counts.values, color=[palette[0], palette[1]])
ax.set_title("Customer Churn Count", color="white")
savefig(fig, "01_churn_count.png")



# 2) Geography vs Exited grouped bar
if "Geography" in df_clean.columns:
    fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
    geo_ct = df_clean.groupby(["Geography", "Exited"]).size().unstack(fill_value=0)
    x = np.arange(len(geo_ct.index)); w = 0.35
    ax.bar(x - w/2, geo_ct[0].values, w, label="Stayed", color=palette[2])
    ax.bar(x + w/2, geo_ct[1].values, w, label="Exited", color=palette[3])
    ax.set_xticks(x); ax.set_xticklabels(geo_ct.index, color="white")
    ax.set_title("Geography vs Churn", color="white"); ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
    savefig(fig, "02_geo_vs_churn.png")



# 3) Gender vs Exited bar
if "Gender" in df_clean.columns:
    fig = plt.figure(figsize=(6,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
    g_ct = df_clean.groupby(["Gender", "Exited"]).size().unstack(fill_value=0)
    ax.bar(g_ct.index.astype(str), g_ct[1].values, color=palette[4])
    ax.set_title("Exits by Gender", color="white"); savefig(fig, "03_gender_exited.png")



# 4) Age distribution histogram
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
ax.hist(df_clean["Age"], bins=20, color=palette[5], edgecolor="white")
ax.set_title("Age Distribution", color="white"); savefig(fig, "04_age_hist.png")



# 5) Age vs Balance scatter (colored by Exited)
fig = plt.figure(figsize=(8,6), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
color_map = df_clean["Exited"].map({0: palette[6], 1: palette[7]})
ax.scatter(df_clean["Age"], df_clean["Balance"], s=12, c=color_map, alpha=0.9)
ax.set_xlabel("Age", color="white"); ax.set_ylabel("Balance", color="white"); ax.set_title("Age vs Balance (by Exited)", color="white")
savefig(fig, "05_age_balance_scatter.png")



# 6) CreditScore histogram
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
ax.hist(df_clean["CreditScore"], bins=25, color=palette[8], edgecolor="white")
ax.set_title("Credit Score Distribution", color="white"); savefig(fig, "06_credit_score_hist.png")



# 7) Tenure counts (bar)
fig = plt.figure(figsize=(10,4), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
ten_ct = df_clean["Tenure"].value_counts().sort_index()
ax.bar(ten_ct.index, ten_ct.values, color=palette[0])
ax.set_title("Customer Count by Tenure", color="white"); savefig(fig, "07_tenure_counts.png")



# 8) NumOfProducts distribution (bar)
fig = plt.figure(figsize=(6,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
nop_ct = df_clean["NumOfProducts"].value_counts().sort_index()
ax.bar(nop_ct.index.astype(str), nop_ct.values, color=palette[1])
ax.set_title("Number of Products Distribution", color="white"); savefig(fig, "08_num_products.png")



# 9) Balance boxplot by Exited
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
b0 = df_clean[df_clean["Exited"]==0]["Balance"]; b1 = df_clean[df_clean["Exited"]==1]["Balance"]
ax.boxplot([b0, b1], labels=["Stayed","Exited"], patch_artist=True, boxprops=dict(facecolor=palette[2], color='white'))
ax.set_title("Balance by Churn (boxplot)", color="white"); savefig(fig, "09_balance_boxplot.png")



# 10) EstimatedSalary histogram
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
ax.hist(df_clean["EstimatedSalary"], bins=25, color=palette[3], edgecolor="white")
ax.set_title("Estimated Salary Distribution", color="white"); savefig(fig, "10_estimated_salary_hist.png")



# 11) Balance vs EstimatedSalary hexbin (dense scatter)
fig = plt.figure(figsize=(8,6), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
hb = ax.hexbin(df_clean["Balance"], df_clean["EstimatedSalary"], gridsize=35, cmap=plt.cm.get_cmap("plasma"))
ax.set_xlabel("Balance", color="white"); ax.set_ylabel("Estimated Salary", color="white"); ax.set_title("Balance vs Estimated Salary (hexbin)", color="white")
cb = fig.colorbar(hb, ax=ax); cb.ax.yaxis.set_tick_params(color="white"); plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
savefig(fig, "11_balance_salary_hexbin.png")



# 12) Correlation matrix (imshow heatmap)
fig = plt.figure(figsize=(8,6), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
corr = df_clean.select_dtypes(include=[np.number]).corr()
cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", color="white"); ax.set_yticklabels(corr.columns, color="white")
ax.set_title("Correlation Matrix", color="white"); cb = fig.colorbar(cax, ax=ax); cb.ax.yaxis.set_tick_params(color="white"); plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
savefig(fig, "12_correlation_heatmap.png")



# 13) CreditScore by Exited (violin)
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
cs0 = df_clean[df_clean["Exited"]==0]["CreditScore"]; cs1 = df_clean[df_clean["Exited"]==1]["CreditScore"]
parts = ax.violinplot([cs0, cs1], showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(palette[4]); pc.set_edgecolor('white')
ax.set_xticks([1,2]); ax.set_xticklabels(["Stayed","Exited"], color="white"); ax.set_title("Credit Score by Churn (violin)", color="white")
savefig(fig, "13_credit_violin.png")



# 14) Age by Exited boxplot
fig = plt.figure(figsize=(8,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
a0 = df_clean[df_clean["Exited"]==0]["Age"]; a1 = df_clean[df_clean["Exited"]==1]["Age"]
ax.boxplot([a0, a1], labels=["Stayed","Exited"], patch_artist=True, boxprops=dict(facecolor=palette[5]))
ax.set_title("Age by Churn (boxplot)", color="white"); savefig(fig, "14_age_boxplot.png")



# 15) IsActiveMember vs Churn (stacked)
fig = plt.figure(figsize=(6,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
iam_ct = df_clean.groupby(["IsActiveMember", "Exited"]).size().unstack(fill_value=0)
x = np.arange(len(iam_ct.index))
ax.bar(x, iam_ct[0].values, label="Stayed", color=palette[6]); ax.bar(x, iam_ct[1].values, bottom=iam_ct[0].values, label="Exited", color=palette[7])
ax.set_xticks(x); ax.set_xticklabels(iam_ct.index.astype(str), color="white"); ax.set_title("IsActiveMember vs Churn (stacked)", color="white"); ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
savefig(fig, "15_active_member_stacked.png")



# 16) HasCrCard vs Exited (simple bar)
fig = plt.figure(figsize=(6,5), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
hcc_ct = df_clean.groupby(["HasCrCard", "Exited"]).size().unstack(fill_value=0)
ax.bar(hcc_ct.index.astype(str), hcc_ct[1].values, color=palette[8])
ax.set_title("Exited by Has Credit Card", color="white"); savefig(fig, "16_hascrcard_exited.png")



# 17) Churn rate by Tenure (line)
fig = plt.figure(figsize=(10,4), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
tenure_rate = df_clean.groupby("Tenure")["Exited"].mean()
ax.plot(tenure_rate.index, tenure_rate.values, marker="o", color=palette[0])
ax.set_title("Churn Rate by Tenure", color="white"); ax.set_xlabel("Tenure (years)", color="white")
savefig(fig, "17_churn_rate_by_tenure.png")



# 18) ROC curve for best model
fig = plt.figure(figsize=(7,6), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
try:
    y_score = best_pipeline.predict_proba(X_test)[:,1]
except Exception:
    y_score = best_pipeline.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color=palette[1], label=f"{best_model_name} (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], linestyle="--", color="white", alpha=0.5)
ax.set_title("ROC Curve - Best Model", color="white"); ax.set_xlabel("False Positive Rate", color="white"); ax.set_ylabel("True Positive Rate", color="white")
ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
savefig(fig, "18_roc_best_model.png")



# 19) Top feature importances (model or permutation)
fig = plt.figure(figsize=(8,6), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    names = feature_names
    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(15, len(names))
    ax.barh(np.array(names)[sorted_idx][:top_n][::-1], importances[sorted_idx][:top_n][::-1], color=palette[2])
    ax.set_title("Top Feature Importances (model)", color="white")
else:
    ax.barh(np.array(feature_names)[perm_idx][:15][::-1], perm.importances_mean[perm_idx][:15][::-1], color=palette[3])
    ax.set_title("Top Feature Importances (permutation)", color="white")
ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
savefig(fig, "19_feature_importance.png")



# 20) Permutation importance full (horizontal)
fig = plt.figure(figsize=(10,8), facecolor="black"); ax = fig.add_subplot(111, facecolor="black")
imp_means = perm.importances_mean
sorted_idx_full = np.argsort(imp_means)[::-1]
ax.barh(np.array(feature_names)[sorted_idx_full][:20][::-1], imp_means[sorted_idx_full][:20][::-1], color=palette[4])
ax.set_title("Permutation Importances (test set)", color="white"); ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
savefig(fig, "20_permutation_importances.png")





# ---------------- Save a CSV with predictions for reference ----------------
pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_pred"] = best_pipeline.predict(X_test)
try:
    pred_df["y_proba"] = best_pipeline.predict_proba(X_test)[:,1]
except Exception:
    pred_df["y_proba"] = np.nan
predictions_csv = os.path.join(output_dir, "predictions_sample.csv")
pred_df.to_csv(predictions_csv, index=False)
print(f"\nSaved predictions sample to: {predictions_csv}")

# ---------- Optional: SHAP explanation (if you have shap installed) ----------
try:
    import shap
    print("SHAP available — creating SHAP summary (may take a moment)...")
    # Use a sample for speed
    sample = X_test_scaled[:500]
    explainer = shap.Explainer(best_model, sample)
    shap_values = explainer(sample)
    # Save SHAP summary (bar)
    shap_fig = plt.figure(figsize=(8,6), facecolor="black")
    shap.summary_plot(shap_values, features=sample, feature_names=feature_names, plot_type="bar", show=False)
    shap_out = os.path.join(output_dir, "shap_summary.png")
    shap_fig.savefig(shap_out, dpi=220, bbox_inches="tight", facecolor=shap_fig.get_facecolor())
    plt.close(shap_fig)
    print(f"Saved SHAP summary to: {shap_out}")
except Exception:
    print("SHAP not installed or failed — skipping SHAP plots. (Optional: pip install shap)")

print("\nAll done. 20 visualizations + predictions saved in:", output_dir)
print("Best model:", best_model_name)









