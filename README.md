# Airbnb Price Prediction (USA)

End-to-end regression project to predict Airbnb listing prices in the USA using
structured listing features (location, room type, availability, reviews) and text
features from listing titles (`name`) via TF-IDF.

This project applies supervised machine learning techniques for price prediction,
including regularized linear models and tree-based ensembles.

Because listing prices are strongly right-skewed, the target is modeled as
`log(price)` for training stability and later evaluated back in USD.

## Objectives
- Perform EDA to understand pricing patterns and outliers
- Build a preprocessing pipeline for mixed data (numeric + categorical + text)
- Train and compare regression models using consistent evaluation
- Select the best model using cross-validation and GridSearchCV

## Dataset
The dataset `AB_US_2023.csv` is included in `data/` and contains Airbnb listings
in the USA with variables such as city, neighbourhood, room type, geo coordinates,
availability, number of reviews, and price.

## Methodology
### 1) Preprocessing & Feature Engineering
- Target transformation: `log(price)`
- Outlier trimming/capping to reduce the influence of extreme listings
- Feature engineering:
  - `state` derived from `city`
  - TF-IDF on `name` (unigrams + bigrams)
  - One-Hot Encoding for categorical variables
  - Standard scaling for numeric variables
- All preprocessing is applied using `ColumnTransformer` + `Pipeline`

### 2) Models Evaluated
- Ridge Regression (GridSearchCV)
- Lasso Regression (GridSearchCV)
- Decision Tree Regressor (GridSearchCV)
- Gradient Boosting Regressor
- XGBoost Regressor

### 3) Metrics
Models are evaluated in both:
- log-space (training objective)
- USD (after inverse transform), reporting RMSE and MAE

## Results (Best Model)
Best model: **XGBoost Regressor**
- RMSE ≈ **85 USD**
- MAE ≈ **58 USD**

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib/seaborn, Jupyter/Colab

## Project Structure

```
airbnb-price-prediction-usa/
├── README.md
├── requirements.txt
├── data/
│ ├── AB_US_2023.csv
│ └── README.md
├── notebooks/
│ └── airbnb_price_prediction.ipynb
└── src/
└── finalairbnb.py
```

## How to Run
1) Install dependencies:
```bash
pip install -r requirements.txt
