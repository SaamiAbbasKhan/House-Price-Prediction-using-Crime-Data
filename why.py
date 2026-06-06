# Why are we building this project: Because we found that crime affect's the house price.
"""
Streamlined analysis for:
- Distributed-lag OLS (ln_crime, ln_crime_lag1, ln_crime_lag2)
- Heterogeneity by area income (proxy via median price)
- Multicollinearity diagnostics (VIF)
- Predictive baseline with MLPRegressor (sklearn)
- Monetary impact translation
- Key plots for presentation
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# statsmodels for econometrics
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# sklearn for ridge, MLP, and preprocessing
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# File paths
DATA_PATH = "dataset.csv"
OUT_DIR = "analysis_outputs"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# ---------- 0) Load data ----------
df = pd.read_csv(DATA_PATH)
# Adjust column names below if your CSV uses different names:
price_col = 'Median'      # median house price
crime_col = 'CrimeCount'  # crime counts
sales_col = 'Sales'       # sales/transactions (optional)
area_col = 'Area'
time_col = 'Date'

# ---------- 1) Preprocessing & feature engineering ----------
# Parse date and create time FE (YearQuarter)
df['Date_parsed'] = pd.to_datetime(df[time_col], errors='coerce')
df['YearQuarter'] = df['Date_parsed'].dt.to_period('Q').astype(str)

# Sort within area and time for lags
df = df.sort_values([area_col, 'Date_parsed']).reset_index(drop=True)

# Log transforms (safe offsets)
df['ln_price'] = np.log(df[price_col].astype(float) + 1e-6)
df['ln_crime'] = np.log(df[crime_col].astype(float) + 1)
if sales_col in df.columns:
    df['ln_sales'] = np.log(df[sales_col].astype(float) + 1)
else:
    df['ln_sales'] = 0.0  # fallback constant so formulas remain valid

# create lags (1,2)
df['ln_crime_lag1'] = df.groupby(area_col)['ln_crime'].shift(1)
df['ln_crime_lag2'] = df.groupby(area_col)['ln_crime'].shift(2)

# Heterogeneity: proxy 'area income' by median price across time per Area
area_medians = df.groupby(area_col)[price_col].median().rename('area_median_price')
df = df.merge(area_medians, left_on=area_col, right_index=True)
median_of_medians = area_medians.median()
df['area_income_group'] = np.where(df['area_median_price'] >= median_of_medians, 'High', 'Low')

# Keep a copy of the prepared data for the main models
df_dl = df.dropna(subset=['ln_price','ln_crime','ln_crime_lag1','ln_crime_lag2','ln_sales','YearQuarter'])


# ---------- 2) Distributed-lag model ----------
# Model: ln_price ~ ln_crime + ln_crime_lag1 + ln_crime_lag2 + ln_sales + Area FE + Quarter FE
controls = " + ln_sales + C({area}) + C(YearQuarter)".format(area=area_col)
formula_dl = "ln_price ~ ln_crime + ln_crime_lag1 + ln_crime_lag2" + controls
model_dl = smf.ols(formula_dl, data=df_dl).fit(cov_type='cluster', cov_kwds={'groups': df_dl[area_col]})

# --- CONSOLE OUTPUT: Print Lag Effects (Coefficients) ---
print("\n--- Distributed-Lag Model Coefficients (Lag Effect) ---")
dl_coeffs_to_print = model_dl.params.loc[['ln_crime', 'ln_crime_lag1', 'ln_crime_lag2', 'ln_sales']]
print(dl_coeffs_to_print)
print("-" * 55)

# ---------- 5) Heterogeneity analysis (setup for plot) ----------
groups = df_dl.groupby('area_income_group')
slopes = {}
for name, g in groups:
    if len(g) >= 5:
        slope = np.polyfit(g['ln_crime'], g['ln_price'], 1)[0]
        slopes[name] = slope
    else:
        slopes[name] = np.nan

# ---------- 6) Multicollinearity diagnostics (VIF) ----------
vif_df = df_dl[['ln_crime','ln_crime_lag1','ln_crime_lag2','ln_sales']].dropna()
X_vif = vif_df.values
vifs = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
vif_table = pd.DataFrame({'variable':['ln_crime','ln_crime_lag1','ln_crime_lag2','ln_sales'],'VIF':vifs})

# --- CONSOLE OUTPUT: Print VIF Table ---
print("\n--- Multicollinearity Diagnostics (VIF) ---")
print(vif_table.to_string(index=False))
print("-" * 43)

if (vif_table['VIF'] > 10).any():
    print("-> High VIF detected. Consider Ridge regression or model simplification.")
else:
    print("-> No severe multicollinearity detected (VIFs <= 10).")

# ---------- 7) Predictive model (MLP) for demonstration ----------
# Build predictive dataset (one-hot area & quarter + ln_crime + ln_sales)
pred_df = df.dropna(subset=['ln_price','ln_crime','ln_sales','YearQuarter'])
X_pred = pd.get_dummies(pred_df[[area_col,'YearQuarter']], drop_first=True)
X_pred['ln_crime'] = pred_df['ln_crime'].values
X_pred['ln_sales'] = pred_df['ln_sales'].values
y_pred = pred_df['ln_price'].values

mlp_pipeline = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), max_iter=800, random_state=42))
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Predictive Model Performance (Mean Squared Error) ---")
try:
    mlp_neg_mse_scores = cross_val_score(mlp_pipeline, X_pred, y_pred, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    print(f"MLP cross-val negative MSE mean: {mlp_neg_mse_scores.mean():.4f}")
except Exception as e:
    print(f"MLP cross-val failed: {e}. Running in-sample fit instead.")
    mlp_pipeline.fit(X_pred, y_pred)
    mse_insample = mean_squared_error(y_pred, mlp_pipeline.predict(X_pred))
    print(f"MLP in-sample MSE: {mse_insample:.4f}")

# Simple linear baseline for comparison
lr = LinearRegression().fit(pred_df[['ln_crime','ln_sales']], y_pred)
mse_lr = mean_squared_error(y_pred, lr.predict(pred_df[['ln_crime','ln_sales']]))
print(f"Simple LR in-sample MSE:          {mse_lr:.4f} (for comparison)")
print("-" * 57)


# ---------- 8) Monetary impact translation ----------
median_price_value = float(df[price_col].median())
# Use the contemporaneous crime coefficient from the distributed lag model
beta_dl_contemporaneous = model_dl.params.get('ln_crime', np.nan)

def pct_from_beta(beta, pct_crime_reduction):
    """Calculates the percentage price change from a log-log model coefficient."""
    # The formula is ( (1 - %reduction)^beta ) - 1
    return (np.exp(beta * np.log(1 - pct_crime_reduction/100)) - 1) * 100

monetary_rows = []
for pct_cut in [5, 10, 20, 50]:
    pct_price_change = pct_from_beta(beta_dl_contemporaneous, pct_cut)
    monetary_rows.append({
        'Crime Reduction (%)': pct_cut,
        'Price Change (%)': pct_price_change,
        'Median Price Before': median_price_value,
        'Median Price After': median_price_value * (1 + pct_price_change / 100),
        'Value Change': median_price_value * (pct_price_change / 100)
    })

monetary_df = pd.DataFrame(monetary_rows)

# --- CONSOLE OUTPUT: Print Monetary Impact Table ---
print("\n--- Monetary Impact of Crime Reduction (based on DL model) ---")
print(monetary_df.to_string(index=False, float_format="%.2f"))
print("-" * 70)


# ---------- 9) Plots for slides ----------
# 9.1 DL numeric coefficients bar chart
coef_names = ['ln_crime','ln_crime_lag1','ln_crime_lag2','ln_sales']
coef_vals = [model_dl.params.get(n, np.nan) for n in coef_names]
plt.figure(figsize=(7,4))
plt.bar(coef_names, coef_vals)
plt.title('Distributed-lag numeric coefficients')
plt.ylabel('Coefficient Value')
plt.axhline(0, color='k', linewidth=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"dl_numeric_coefs.png"))
plt.close()

# 9.2 Residuals histogram for DL model
plt.figure(figsize=(6,4))
plt.hist(model_dl.resid, bins=40, edgecolor='k')
plt.title('DL model residuals')
plt.xlabel('Residual Value')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"dl_residuals_hist.png"))
plt.close()

# 9.3 Heterogeneity slopes (visual)
plt.figure(figsize=(5,4))
plt.bar(list(slopes.keys()), list(slopes.values()))
plt.title('Visual slope: ln_price vs ln_crime by area income')
plt.ylabel('Simple Regression Slope')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"heterogeneity_slopes_vis.png"))
plt.close()

# 9.4 Area-level horizontal bar sorted by mean crime
area_summary = df.groupby(area_col).agg(area_mean_crime=(crime_col,'mean')).reset_index()
area_summary = area_summary.sort_values('area_mean_crime', ascending=False)
plt.figure(figsize=(7, max(4, 0.2*len(area_summary))))
plt.barh(area_summary[area_col], area_summary['area_mean_crime'])
plt.xlabel('Mean Crime Count per Quarter')
plt.ylabel('Area')
plt.title('Area Mean Crime Levels')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"area_mean_crime_barh.png"))
plt.close()


print(f"\nAnalysis complete. Requested plots saved to folder: {OUT_DIR}")