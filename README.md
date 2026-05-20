# Predict Quantity Sold

Predicts `Quantity_Sold_(kilo)` for supermarket sales data using a machine learning pipeline.

## Problem
Supermarkets need to forecast how much of each product will sell to manage 
inventory and reduce waste. This project builds a regression model to predict 
the quantity sold (in kilograms) based on pricing, product, and time features.

## Data
- `labeled_data.csv` — historical sales with known quantities
- `unlabeled_data.csv` — sales records without quantities (for final prediction)

Key features: item name, category, unit selling price, wholesale price, 
loss rate, weekday.

## Approach

**Preprocessing**
- Removed negative prices using absolute value
- Filtered out rows with negative target values
- Filled missing wholesale prices with training set mean

**Feature Engineering**
- Price Difference: selling price minus wholesale price
- Profit Ratio: selling price divided by wholesale price
- Loss Adjusted Margin: effective margin after accounting for spoilage

**Encoding**
- Item name and category: target encoding (replaced with mean quantity sold per category)
- Weekday: one-hot encoding

**Models Trained**
- Linear Regression (baseline comparison)
- Random Forest Regressor (final model)

## Results

| Model | Val R² | Val MSE | Baseline MSE |
|-------|--------|---------|--------------|
| Linear Regression | — | — | 165.11 |
| Random Forest | 0.61 | 63.80 | 165.11 |

- Train R²: 0.80 vs Val R²: 0.61 — mild overfitting present
- Model significantly outperforms naive baseline (predict mean)
- Most predictive feature: Item identity (0.36 importance score)

## What I'd Improve Next
- Reduce overfitting: lower max_depth, tune with cross-validation 
  instead of a single split
- Replace mean imputation for wholesale price with a more principled 
  approach — check if missingness is systematic
- Target encoding without CV folds risks subtle data leakage — 
  use category_encoders with proper pipeline next time
- Investigate and act on outliers instead of just plotting them
- Try log-transforming the target to handle skewed distribution

## Setup

```bash
git clone https://github.com/215018/Predict-the-quantity-sold-in-the-supermarket
cd Predict-the-quantity-sold-in-the-supermarket
python predict-the-quantitysold.py
```

## Output
`submission.csv` — predicted quantities for the unlabeled dataset
