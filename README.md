# 🏠 London House Price Prediction using Machine Learning, Deep Learning & Crime Data

## Overview

This project investigates the relationship between crime rates and housing prices across London boroughs while building a highly accurate machine learning system capable of predicting house prices using historical housing and crime data.

The project began as an econometric research study to answer a fundamental question:

> **Does crime significantly affect property values?**

After establishing that crime is indeed a significant predictor of housing prices through statistical and econometric analysis, the project evolved into a full-scale machine learning pipeline capable of predicting house prices with high accuracy using advanced ML and Deep Learning techniques.

The final predictive system achieved **97.14% prediction accuracy**, with **XGBoost emerging as the best-performing model**.

---

## Project Evolution

This repository contains two major stages of the project:

### 1. why.py — Research & Validation Stage

Before building a predictive model, it was necessary to determine whether crime actually influences housing prices.

This phase focused on:

* Distributed-Lag Regression
* First-Difference Regression
* Placebo Testing
* Heterogeneity Analysis
* Neural Network Verification (MLP)

The objective was to establish a statistically valid relationship between crime rates and property values.

### Key Finding

A 1% increase in crime resulted in an approximate cumulative decrease of **0.135% in house prices over three quarters**, demonstrating that crime is a significant factor in housing valuation.

Once the hypothesis was validated, the project moved to the predictive modeling stage.

---

### 2. main.py — Advanced House Price Prediction System

The second phase focused on building a production-style machine learning pipeline capable of accurately predicting London house prices.

Multiple machine learning and deep learning models were trained and compared:

* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* Hybrid Deep Learning Network (LSTM + MLP)

The objective was not only to maximize prediction accuracy but also to understand which factors drive property values.

---



### 3. dataset_merging/ — Dataset Preparation & Integration Stage

> **Note:** There is no need to run anything in this folder for the main project. The final cleaned dataset is already available in the project root and is directly used by `why.py` and `main.py`.
>
> This folder is included to demonstrate how the final dataset was constructed from the original raw datasets. Run the scripts here if you want to understand the complete data preparation and merging process.

This folder contains:

- `land-registry-house-prices-borough (1).csv` — Raw London borough-level house price dataset.
- `MPS Borough Level Crime (Historical).csv` — Raw London borough-level crime dataset.
- `datasetMerging.py` — Data preprocessing and dataset integration pipeline.
- 
---

# Dataset

The project combines two independent datasets:

### London Borough House Price Dataset

Contains borough-level housing statistics including:

* Mean House Price
* Median House Price
* Sales Volume
* Borough Information
* Historical Trends

### London Crime Dataset

Contains:

* Borough-wise Crime Counts
* Historical Crime Statistics
* Temporal Crime Trends

---

## Data Engineering & Integration

The original datasets had different structures:

* One dataset was stored in a **wide format**
* One dataset was stored in a **long format**

Significant preprocessing was required before analysis could begin.

### Data Preparation Steps

#### Data Cleaning

* Removed missing values
* Standardized borough names
* Corrected inconsistent records
* Handled duplicate entries

#### Dataset Transformation

* Converted incompatible formats
* Restructured long and wide datasets
* Aligned temporal records
* Created a unified analytical dataset

#### Data Merging

Using Pandas, both datasets were merged into a single borough-level time-series dataset.

#### Outlier Handling

Several abnormal observations were identified and removed to improve model robustness.

#### Feature Engineering

Additional intelligent features were created, including:

##### Lag Features

Historical crime information:

* CrimeCount_lag1
* CrimeCount_lag2

##### Rolling Features

Long-term neighborhood safety trends:

* CrimeCount_roll_avg4

##### Temporal Features

* Year
* Quarter
* Historical trends

These engineered features significantly improved predictive performance.

---

# Technologies Used

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras

### Statistical Analysis

* Econometrics
* Regression Analysis
* Fixed Effects Modeling

---

# Exploratory Data Analysis

Several visual analyses were conducted:

### House Price vs Crime Trends

Analyzed long-term relationships between:

* House prices
* Crime counts
* Sales volume

### Correlation Analysis

A correlation matrix revealed:

* Strong positive correlation between year and prices
* Negative relationship between crime and house prices

This provided initial evidence supporting the project's hypothesis.

---

# Models Implemented

## Decision Tree Regressor

Baseline model used for comparison.

### Advantages

* Easy to interpret
* Fast training

---

## Random Forest Regressor

Ensemble model consisting of hundreds of decision trees.

### Advantages

* Reduced overfitting
* Strong predictive performance
* High robustness

---

## XGBoost Regressor

State-of-the-art gradient boosting model.

### Advantages

* Sequential error correction
* Excellent performance on tabular data
* Industry-standard predictive algorithm

### Result

🏆 Best performing model

* Prediction Accuracy: **97.14%**
* R² Score: **99.45%**
* Mean Absolute Error: **£14,803**

---

## Hybrid Deep Learning Network

A custom neural network architecture was developed to capture both temporal and static information.

### Architecture

#### LSTM Branch

Captures:

* Time-series behavior
* Historical crime trends
* Market evolution

#### MLP Branch

Captures:

* Borough-specific information
* Static characteristics

Both branches are merged before generating the final prediction.

This architecture allows the model to simultaneously learn:

* Temporal relationships
* Spatial relationships

---

# Research Findings

The project provides strong evidence that crime influences housing prices.

### Distributed Lag Results

A 1% increase in crime produces:

* Immediate impact: -0.067%
* One-quarter lag effect: -0.038%
* Two-quarter lag effect: -0.030%

### Total Effect

A 1% increase in crime is associated with approximately:

**-0.135% reduction in house prices over three quarters**

---

# Feature Importance Analysis

The most influential features were:

1. Year
2. Borough Identity
3. CrimeCount_roll_avg4
4. CrimeCount_lag1
5. Sales Volume

The analysis confirms that both recent and long-term crime trends are important predictors of housing values.

---

# Project Workflow

```text
Raw Housing Data
        +
Raw Crime Data
        ↓

Data Cleaning
        ↓

Transformation
(Long ↔ Wide Format Handling)
        ↓

Dataset Integration
        ↓

Outlier Removal
        ↓

Feature Engineering
        ↓

Exploratory Data Analysis
        ↓

Research Validation (why.py)
        ↓

Predictive Modeling (main.py)
        ↓

Model Comparison
        ↓

Interpretation & Insights
```

---

# Future Improvements

Potential future upgrades include:

### Hyperparameter Optimization

* GridSearchCV
* Bayesian Optimization
* Optuna

### Explainable AI

Using SHAP values to explain:

* Individual predictions
* Crime contribution to house prices
* Borough-level decision patterns

### Additional Features

* Interest Rates
* Inflation Data
* School Ratings
* Transportation Access
* Geospatial Features

---

# Results Summary

| Model                 | Performance |
| --------------------- | ----------- |
| Decision Tree         | Baseline    |
| Random Forest         | Strong      |
| Hybrid Neural Network | Excellent   |
| XGBoost               | 🏆 Best     |

### Final Performance

* Prediction Accuracy: **97.14%**
* R² Score: **99.45%**
* MAE: **£14,803**

---

# Conclusion

This project successfully combines econometric research, machine learning, and deep learning to understand and predict London housing prices.

The research phase demonstrated that crime has a statistically significant negative effect on property values. The predictive phase then leveraged these insights to build a highly accurate house price prediction system.

The final XGBoost model achieved **97.14% prediction accuracy**, while feature importance analysis confirmed that both recent and long-term crime trends play a critical role in determining London house prices.
