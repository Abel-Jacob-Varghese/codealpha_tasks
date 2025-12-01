import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Setup paths ----------
current_dir = os.path.dirname(__file__)  # folder where main.py is located
data_filename_candidates = ["Advertising.csv", "advertising.csv", "Advertising data.csv", "Advertising.csv.zip"]

# find file in folder
data_path = None
for f in os.listdir(current_dir):
    if f.lower().startswith("advert") and f.lower().endswith(".csv"):
        data_path = os.path.join(current_dir, f)
        break

if data_path is None:
    raise FileNotFoundError("Advertising CSV not found in the script folder. Put the dataset (Advertising.csv) in the same folder as this script.")

plots_dir = os.path.join(current_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# ---------- Load dataset ----------
data = pd.read_csv(data_path)
print("✅ Dataset loaded:", os.path.basename(data_path))
print("Shape:", data.shape)

# ---------- Quick cleaning & column mapping ----------
# Normalize column names (strip, lower)
data.columns = [c.strip() for c in data.columns]

# Map common column name variations to standard names
col_map = {}
cols_lower = [c.lower() for c in data.columns]
# possible columns: tv, radio, newspaper, sales, advertising, spend, platform, target
if 'tv' in cols_lower:
    col_map[data.columns[cols_lower.index('tv')]] = 'TV'
if 'radio' in cols_lower:
    col_map[data.columns[cols_lower.index('radio')]] = 'Radio'
if 'newspaper' in cols_lower:
    col_map[data.columns[cols_lower.index('newspaper')]] = 'Newspaper'
if 'sales' in cols_lower:
    col_map[data.columns[cols_lower.index('sales')]] = 'Sales'
# fallback: try guessing by contains
for c in data.columns:
    cl = c.lower()
    if 'spend' in cl or 'advert' in cl and 'tv' not in cols_lower:
        # if the set doesn't have TV/Radio/Newspaper, just print columns
        pass

data = data.rename(columns=col_map)

print("\nColumns after mapping:", data.columns.tolist())

# Check required columns exist
required = ['Sales']
features_candidates = ['TV', 'Radio', 'Newspaper']  # typical
for r in required:
    if r not in data.columns:
        raise ValueError(f"Required column '{r}' not found in dataset. Found columns: {data.columns.tolist()}")

# If feature columns not present, use all numeric columns except Sales
features_present = [f for f in features_candidates if f in data.columns]
if not features_present:
    # choose numeric cols except Sales
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    features_present = [c for c in numeric_cols if c != 'Sales']

print("Features used for modelling:", features_present)

# ---------- Handle missing values ----------
print("\nMissing values per column:\n", data.isnull().sum())
# Simple strategy: drop rows with missing target or features
data = data.dropna(subset=['Sales'])
data = data.dropna(subset=features_present)
data = data.reset_index(drop=True)

# ---------- Save cleaned data for Power BI ----------
cleaned_path = os.path.join(current_dir, "cleaned_sales_data.csv")
data.to_csv(cleaned_path, index=False)
print("\n💾 Cleaned dataset saved as:", cleaned_path)

# ---------- Exploratory Data Analysis (plots saved) ----------
sns.set(style="whitegrid")

# Correlation heatmap (only numeric)
plt.figure(figsize=(6,5))
num_cols = data.select_dtypes(include=np.number).columns.tolist()
sns.heatmap(data[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
plt.close()

# Pairplot (scatter matrix) - save as PNG (may be heavy for many points)
try:
    sns.pairplot(data[features_present + ['Sales']])
    plt.savefig(os.path.join(plots_dir, "pairplot_features_sales.png"))
    plt.close()
except Exception:
    # fallback if pairplot fails in headless environment
    pass

# Feature vs Sales scatter plots
for feat in features_present:
    plt.figure(figsize=(7,4))
    sns.scatterplot(x=data[feat], y=data['Sales'], alpha=0.7)
    plt.xlabel(feat)
    plt.ylabel("Sales")
    plt.title(f"{feat} vs Sales")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{feat}_vs_sales.png"))
    plt.close()

# ---------- Prepare feature matrix and target ----------
X = data[features_present].copy()
y = data['Sales'].copy()

# optional: log-transform if skewed - we keep raw for interpretability
# X = X.apply(np.log1p)  # commented out; enable if you want

# ---------- Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Train models ----------
lr = LinearRegression()
lr.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ---------- Predictions ----------
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)

# ---------- Evaluation function ----------
def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

mae_lr, mse_lr, rmse_lr, r2_lr = eval_metrics(y_test, y_pred_lr)
mae_rf, mse_rf, rmse_rf, r2_rf = eval_metrics(y_test, y_pred_rf)

print("\n📊 Model Performance on Test Set:")
print(f"Linear Regression -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}, R²: {r2_lr:.3f}")
print(f"Random Forest     -> MAE: {mae_rf:.3f}, RMSE: {rmse_rf:.3f}, R²: {r2_rf:.3f}")

# ---------- Cross-validation (5-fold) ----------
cv_lr = cross_val_score(lr, X, y, cv=5, scoring='r2')
cv_rf = cross_val_score(rf, X, y, cv=5, scoring='r2')
print("\n📈 Cross-validation R² (5-fold):")
print(f"Linear Regression CV R² mean: {cv_lr.mean():.3f} (std {cv_lr.std():.3f})")
print(f"Random Forest CV R² mean:     {cv_rf.mean():.3f} (std {cv_rf.std():.3f})")

# ---------- Save actual vs predicted plot (Random Forest) ----------
plt.figure(figsize=(7,6))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "actual_vs_predicted_rf.png"))
plt.close()

# ---------- Residuals histogram ----------
residuals = y_test - y_pred_rf
plt.figure(figsize=(7,4))
sns.histplot(residuals, kde=True)
plt.title("Residuals (RF): Actual - Predicted")
plt.xlabel("Residual")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "residuals_rf.png"))
plt.close()

# ---------- Feature importance (RF) ----------
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x='Importance', y='Feature', data=feat_imp)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "feature_importance_rf.png"))
plt.close()

# ---------- Linear regression coefficients ----------
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df)
plt.title("Linear Regression Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "linear_coefficients.png"))
plt.close()

# ---------- Save predictions to csv for Power BI / review ----------
preds_df = X_test.copy()
preds_df['Actual_Sales'] = y_test.values
preds_df['Predicted_Sales_RF'] = y_pred_rf
preds_df['Predicted_Sales_LR'] = y_pred_lr
preds_out_path = os.path.join(current_dir, "sales_predictions_vs_actuals.csv")
preds_df.to_csv(preds_out_path, index=False)

print("\n💾 Predictions saved to:", preds_out_path)
print("✅ All plots saved to:", plots_dir)
print("\n🔍 Quick Insights:")
print("- Check 'feature_importance_rf.png' to see which ad channels matter most")
print("- If TV/Radio show high importance, they are strong predictors of Sales")
print("- Compare LR coefficients vs RF importance to understand linear vs non-linear relationships")
print("\n✅ Task 4 completed successfully!")
