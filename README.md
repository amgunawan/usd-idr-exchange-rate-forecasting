# **Analysis of Transformer, LSTM, and XGBoost Performance in Predicting the USD/IDR Exchange Rate**

This project compares the forecasting performance of three models — **Transformer**, **LSTM**, and **XGBoost** — on predicting the daily USD/IDR exchange rate using historical data.

## Objective

Evaluate and compare Transformer, LSTM, and XGBoost on their ability to forecast the USD/IDR exchange rate, assessing prediction accuracy and how well each model captures the underlying exchange rate patterns.

## Models

| Model           | Description                                                                                                               |
|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| **Transformer** | Self-attention based deep learning model, designed to capture long-range dependencies and complex temporal relationships. |
| **LSTM**        | Recurrent neural network architecture built to learn long-term dependencies in sequential data.                           |
| **XGBoost**     | Gradient-boosted decision tree ensemble, applied here with engineered lag features.                                       |

## Dataset

- **Source:** [Bank Indonesia — USD/IDR Exchange Rate](https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/Default.aspx)
- **Target:** USD → IDR daily official exchange rate
- **Period:** 1 June 2016 – 30 June 2026 (10 years)
- **File:** `Kurs Jisdor 2016-2026.xlsx`ss

## How to Run

This project requires Python 3.11 (recommended for best compatibility with TensorFlow, PyTorch, and XGBoost on Apple Silicon) and a clean virtual environment.

### 1. Clone the repository

```bash
git clone git@github.com:amgunawan/usd-idr-exchange-rate-forecasting
```

### 3. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

The path should point inside the `venv` folder, not a system or Anaconda Python.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Register the venv as a Jupyter kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

### 6. Open the notebook

- Open `data-analysis.ipynb` in VS Code (or Jupyter).
- Select the kernel **"Python (venv)"** from the kernel picker (top-right of the notebook).
- Run cells from top to bottom, or use **Run All**.

## Workflow

1. **Data Preparation**
   - Drop unused columns, parse dates, sort chronologically
   - Check for duplicates and missing values
   - Reindex to a full daily calendar and forward-fill non-trading days (weekends/holidays)
   - Filter to the 2016-06-01 to 2026-06-30 analysis window

2. **Exploratory Data Analysis**
   - Time series plot, seasonal decomposition (trend/seasonal/residual)
   - Distribution (histogram, boxplot) and monthly seasonality by year

3. **Train/Test Split**
   - Chronological 80/20 split (no shuffling, to preserve temporal order)

4. **Normalization**
   - `MinMaxScaler` fit on train, applied to test

5. **Window Size Selection**
   - ACF/PACF analysis to inform the sliding window length

6. **Sliding Window Sequences**
   - 90-day lookback window used to predict the next day's rate
   - Reshaped appropriately per model (3D sequences for Transformer/LSTM, flattened 2D for XGBoost)

7. **Modeling**
   - **Transformer** (PyTorch): custom encoder with positional encoding, `d_model=64`, `nhead=4`, `num_layers=2`, dropout 0.1, trained with Adam (lr=0.001), MSE loss, early stopping (patience 10), best checkpoint saved to `weight/best_transformer.pth`
   - **LSTM** (Keras): 2 stacked LSTM layers (64 units each) with dropout 0.1, dense output layer, Adam (lr=0.001), MSE loss, early stopping (patience 10, restores best weights)
   - **XGBoost**: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping (10 rounds)

8. **Evaluation**
   - Metrics: MAE, RMSE, MAPE, R²
   - Predictions and actuals inverse-transformed back to original scale before scoring

## Results

| Model       | MAE       | RMSE      | MAPE (%) | R²         |
|-------------|-----------|-----------|----------|------------|
| Transformer | 108.82    | 130.9     | 0.66     | 0.9445     |
| **LSTM**    | **59.23** | **76.81** | **0.36** | **0.9809** |
| XGBoost     | 339.91    | 530.35    | 1.98     | 0.0894     |

## Conclusion

**LSTM is the best-performing model**, achieving the lowest error on every metric and explaining about 98.1% of the variance in the actual exchange rate. Transformer ranks second: it follows the main upward trend well, but its MAE and RMSE are roughly 1.8× and 1.7× those of LSTM, and its forecasts are smoother than the actual series. They overestimate during the mid-test decline and miss the sharp short-term swings near the end. XGBoost performs worst: it cannot extrapolate beyond the range seen in training, so its predictions flatten once the exchange rate moves into new levels.