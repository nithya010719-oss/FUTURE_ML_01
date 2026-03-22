# Store Sales — Time Series Forecasting

A machine learning solution for the [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

The goal is to predict unit sales for 54 stores across 33 product families over a 15-day forecast horizon (August 16–31, 2017), using over 4.5 years of historical transaction data.

---

## Project Structure

```
├── store_sales_forecasting.ipynb   # Main notebook — EDA, model training, predictions
├── submission.csv                  # Final predictions (28,512 rows)
├── Store_Sales_Forecasting_Whitepaper.docx  # Full project whitepaper with charts
└── README.md
```

> **Dataset files** (`train.csv`, `test.csv`) are not included. Download them from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and place them in the same directory before running the notebook.

---

## Problem Statement

- **Task:** Regression — predict daily unit sales per store-product-family combination
- **Training data:** January 1, 2013 to August 15, 2017 (3,000,888 rows)
- **Test data:** August 16, 2017 to August 31, 2017 (28,512 rows)
- **Evaluation metric:** Root Mean Squared Log Error (RMSLE)
- **Stores:** 54 | **Product families:** 33

---

## Approach

### Exploratory Data Analysis
- Monthly sales trend analysis (consistent year-over-year growth)
- Top product families by revenue (GROCERY I, BEVERAGES, PRODUCE dominate at ~68%)
- Day-of-week patterns (weekends 20–25% higher than weekdays)
- Promotion impact analysis (7x sales lift on promoted days)

### Feature Engineering

| Feature | Description |
|---|---|
| `lag_16`, `lag_21`, `lag_28` | Lagged sales values (minimum 16-day offset to prevent data leakage) |
| `rolling_7d`, `rolling_14d` | Rolling averages anchored at lag_16 |
| `day_of_week`, `month`, `year` | Calendar features for seasonality and trend |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `promo` | log1p-transformed promotion count |
| `family_code` | Integer-encoded product family |
| `store_nbr` | Store identifier |

**Target transformation:** `log1p(sales)` applied before training; `expm1` applied after prediction to recover original scale.

### Model

**Random Forest Regressor** (scikit-learn)

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)
```

- Trained on a 250,000-row sample from 2015–2017 for memory efficiency
- Validation split: last 15 days of training data (August 1–15, 2017)

---

## Validation Results

Evaluated on 26,730 samples (August 1–15, 2017):

| Metric | Score |
|---|---|
| RMSLE | 0.4498 |
| MAE | 77.0 units |
| R² | 0.951 |

The model explains **95.1% of the variance** in sales. Predictions are off by ~77 units on average per store-product-day, which is well within an acceptable range given that high-volume categories such as GROCERY I record thousands of units per day.

---

## Forecast Highlights

| Metric | Value |
|---|---|
| Total forecast (15 days) | 13.17 million units |
| Average daily sales | ~823,000 units across all 54 stores |
| Year-over-year growth | +15.9% vs August 2016 |
| Promotion lift | 7x vs non-promotion days |
| Busiest forecast day | Sunday, August 20 |

---

## Requirements

```
python >= 3.8
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

Install dependencies:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn jupyter
```

---

## How to Run

1. Clone the repository and download the dataset from Kaggle:
   ```bash
   git clone https://github.com/nithya010719-oss/FUTURE_ML_01.git
   cd FUTURE_ML_01
   ```

2. Place `train.csv` and `test.csv` in the project directory.

3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook store_sales_forecasting.ipynb
   ```

4. Run all cells. The notebook will generate:
   - All EDA and forecast visualisation charts
   - A trained Random Forest model
   - `submission.csv` with 28,512 predictions

---

## Deliverables

- **`store_sales_forecasting.ipynb`** — Complete notebook covering EDA, feature engineering, model training, validation, and forecast visualisations
- **`submission.csv`** — Kaggle submission file with predictions for all store-product-day combinations
- **`Store_Sales_Forecasting_Whitepaper.docx`** — Professional whitepaper explaining the methodology, EDA findings, forecast results, and business recommendations

---

## References

*Dataset source: [Kaggle — Store Sales Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)*
*Task: Future Interns ML Track — Task 01*

---

## Author

**Jagruthi Nithya** — Future Interns ML Intern
GitHub: [@nithya010719-oss](https://github.com/nithya010719-oss)


