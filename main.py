# ==============================================================================
# COMPREHENSIVE HOUSE PRICE PREDICTION
# ==============================================================================
# This is the definitive version of the project. It builds and evaluates four
# key machine learning models, including an advanced hybrid neural network.
# It automatically saves all graphs to a 'visuals' folder AND displays them
# live upon execution. It provides a clear, intuitive 'Prediction Accuracy' score.

print("PHASE 0: Initializing Project...")

# --- Library Imports ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# --- Setup Output Folder ---
output_dir = 'visuals'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"✅ All graphs will be saved to the '{output_dir}/' folder.")

# --- Plotting Style ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# ==============================================================================
# PHASE 1: DATA LOADING & EXPLORATORY ANALYSIS
# ==============================================================================
print("\nPHASE 1: Loading Data and Performing EDA...")

df = pd.read_csv('dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# --- Generate and Save EDA Visuals ---
print("Generating and saving EDA visualizations...")

# Plot 1: Overall Trends
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
df.groupby('Date')['Mean'].mean().plot(ax=ax1, color='b', lw=2, label='Avg. House Price')
df.groupby('Date')['CrimeCount'].sum().plot(ax=ax2, color='r', linestyle='--', lw=2, label='Total Crime Count')
ax1.set_xlabel('Date')
ax1.set_ylabel('Average House Price (£)', color='b')
ax2.set_ylabel('Total Crime Count', color='r')
ax1.set_title('London House Prices vs. Crime Count Over Time', fontsize=16)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, '01_price_vs_crime_trends.png'))
plt.show() # Display the plot
plt.close()

# Plot 2: Correlation Matrix Heatmap
# Create temporary numerical features for correlation analysis
df_corr = df.copy()
df_corr['Year'] = df_corr['Date'].dt.year
df_corr['Month'] = df_corr['Date'].dt.month
correlation_matrix = df_corr[['Mean', 'Median', 'Sales', 'CrimeCount', 'Year']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Features', fontsize=16)
plt.savefig(os.path.join(output_dir, '02_correlation_matrix.png'))
plt.show() # Display the plot
plt.close()


# ==============================================================================
# PHASE 2: ADVANCED FEATURE ENGINEERING
# ==============================================================================
print("\nPHASE 2: Engineering Time-Aware Features...")

df = df.sort_values(by=['Area', 'Date']).reset_index(drop=True)
df['CrimeCount_lag1'] = df.groupby('Area')['CrimeCount'].shift(1)
df['Sales_lag1'] = df.groupby('Area')['Sales'].shift(1)
df['CrimeCount_roll_avg4'] = df.groupby('Area')['CrimeCount'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df.dropna(inplace=True)
print("✅ Feature engineering complete.")

# ==============================================================================
# PHASE 3: DATA SPLITTING & PREPROCESSING
# ==============================================================================
print("\nPHASE 3: Preparing Data for Models...")

df['Mean_log'] = np.log1p(df['Mean'])
target = 'Mean_log'
numerical_features = ['Sales', 'CrimeCount', 'Year', 'Quarter', 'CrimeCount_lag1', 'Sales_lag1', 'CrimeCount_roll_avg4']
categorical_features = ['Area']

X = df[numerical_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split into {len(X_train)} training and {len(X_test)} testing samples.")

# Universal preprocessor for all tree-based models
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# ==============================================================================
# PHASE 4: MODEL BUILDING & TRAINING
# ==============================================================================
print("\nPHASE 4: Building and Training All Models...")

# --- Model 1: Decision Tree (Baseline) ---
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(random_state=42))])
dt_pipeline.fit(X_train, y_train)
print("✅ Decision Tree training complete.")

# --- Model 2: Random Forest (Standard Ensemble) ---
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
rf_pipeline.fit(X_train, y_train)
print("✅ Random Forest training complete.")

# --- Model 3: XGBoost (Advanced Ensemble) ---
X_train_processed_xgb = preprocessor.fit_transform(X_train)
X_test_processed_xgb = preprocessor.transform(X_test)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8,
                             random_state=42, n_jobs=-1, early_stopping_rounds=50)
xgb_model.fit(X_train_processed_xgb, y_train, eval_set=[(X_test_processed_xgb, y_test)], verbose=False)
print("✅ XGBoost training complete.")

# --- Model 4: Hybrid Deep Learning (Advanced Neural Network) ---
ohe_nn = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_nn = ohe_nn.fit_transform(X_train[categorical_features])
X_test_cat_nn = ohe_nn.transform(X_test[categorical_features])
scaler_nn = StandardScaler()
X_train_num_scaled_nn = scaler_nn.fit_transform(X_train[numerical_features])
X_test_num_scaled_nn = scaler_nn.transform(X_test[numerical_features])
X_train_lstm = X_train_num_scaled_nn.reshape((X_train_num_scaled_nn.shape[0], 1, X_train_num_scaled_nn.shape[1]))
X_test_lstm = X_test_num_scaled_nn.reshape((X_test_num_scaled_nn.shape[0], 1, X_test_num_scaled_nn.shape[1]))

lstm_input = Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), name='lstm_input')
lstm_branch = LSTM(64, activation='relu')(lstm_input)
mlp_input = Input(shape=(X_train_cat_nn.shape[1],), name='mlp_input')
mlp_branch = Dense(32, activation='relu')(mlp_input)
combined = Concatenate()([lstm_branch, mlp_branch])
final_dense = Dense(64, activation='relu')(combined)
output = Dense(1, name='output')(final_dense)

hybrid_nn_model = Model(inputs=[lstm_input, mlp_input], outputs=output)
hybrid_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = hybrid_nn_model.fit([X_train_lstm, X_train_cat_nn], y_train, epochs=200, batch_size=32,
                              validation_split=0.2, callbacks=[early_stopping_callback], verbose=0)
print("✅ Hybrid Deep Learning training complete.")

# Visualize and Save Training History
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Deep Learning Model Training Process ("The Heartbeat")', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Loss (Error)')
plt.legend()
plt.savefig(os.path.join(output_dir, '03_deep_learning_training_history.png'))
plt.show()
plt.close()
print("✅ Saved and displayed Deep Learning training history graph.")

# ==============================================================================
# PHASE 5: FINAL EVALUATION WITH ACCURACY SCORE
# ==============================================================================
print("\nPHASE 5: Evaluating Final Performance...")

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rsquared = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape
    return mae, rsquared, accuracy

# Get predictions for all models
y_pred_dt_log = dt_pipeline.predict(X_test)
y_pred_rf_log = rf_pipeline.predict(X_test)
y_pred_xgb_log = xgb_model.predict(X_test_processed_xgb)
y_pred_nn_log = hybrid_nn_model.predict([X_test_lstm, X_test_cat_nn], verbose=0).flatten()

# Convert all predictions back to original price scale
y_true_price = np.expm1(y_test)
predictions = {
    "Decision Tree": np.expm1(y_pred_dt_log),
    "Random Forest": np.expm1(y_pred_rf_log),
    "XGBoost": np.expm1(y_pred_xgb_log),
    "Hybrid Deep Learning": np.expm1(y_pred_nn_log)
}

# Calculate and store results
results_list = []
for name, y_pred in predictions.items():
    mae, r2, acc = calculate_metrics(y_true_price, y_pred)
    results_list.append({
        'Model': name,
        'MAE (Avg. Error)': f"£{mae:,.0f}",
        'R-squared': f"{r2:.2%}",
        'Prediction Accuracy': f"{acc:.2f}%"
    })

results_df = pd.DataFrame(results_list).sort_values(by='MAE (Avg. Error)')

print("\n--- Final Model Performance Comparison ---")
print(results_df.to_string(index=False))

# --- Visualize Predictions of the Best Model ---
best_model_name = results_df.iloc[0]['Model']
best_predictions = predictions[best_model_name]

plt.figure(figsize=(10, 8))
plt.scatter(y_true_price, best_predictions, alpha=0.6, edgecolors='k', label='Model Predictions')
plt.plot([y_true_price.min(), y_true_price.max()], [y_true_price.min(), y_true_price.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual House Prices (£)", fontsize=12)
plt.ylabel("Predicted House Prices (£)", fontsize=12)
plt.title(f"Best Model ({best_model_name}): Actual vs. Predicted Prices", fontsize=16)
plt.legend()
plt.savefig(os.path.join(output_dir, '04_best_model_predictions.png'))
plt.show()
plt.close()
print(f"✅ Saved and displayed prediction plot for the best model ({best_model_name}).")

# ==============================================================================
# PHASE 6: FEATURE IMPORTANCE (THE 'WHY')
# ==============================================================================
print("\nPHASE 6: Analyzing What Drives the Predictions...")

# Use the fitted preprocessor from one of the pipelines to get feature names
feature_names = numerical_features + list(rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
# FIX: Correct syntax for Seaborn palette warning
sns.barplot(x='importance', y='feature', data=importance_df.head(15), hue='feature', palette='inferno', dodge=False, legend=False)
plt.title("Top 15 Most Important Features for Price Prediction (from XGBoost)", fontsize=16)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig(os.path.join(output_dir, '05_feature_importance.png'))
plt.show()
plt.close()
print("✅ Saved and displayed feature importance graph.")

print("\n\nProject execution complete. Everything is ready.")