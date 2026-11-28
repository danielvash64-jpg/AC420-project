# Future ROA Predictor

A Streamlit app that forecasts a company's next-year return on assets (ROA) using regression models built with **forward selection**, one per GICS industry group (2010, 2550, 3520, 4510, 4530). Type a stock ticker and the app pulls the latest annual financial statements from Yahoo Finance (via `yfinance`), averages each ratio over the most recent (up to) 5 fiscal years, and forecasts ROA for the next fiscal year with a 95% range of forecast ± 2 standard errors.

## Repository layout

```
app.py                          Streamlit app (entry point)
requirements.txt                Packages for Streamlit Community Cloud
data/complete_companies_2014-2025.xlsx   Training data (Compustat annual, 2014-2025)
training/
  roa_model.py                  Variable construction, forward selection, reporting
  train_2010.py ... train_4530.py   One training script per industry group
  requirements-training.txt     Extra packages needed to run training (statsmodels, scipy, ...)
  results/
    results_<group>.md          Step-by-step forward selection report (renders on GitHub)
    model_<group>.json          Final coefficients, covariance matrix, bounds (used by app.py)
```

## Method

**Target:** next fiscal year ROA, where ROA = net income / average total assets.

**Candidate variables (each averaged over 5 fiscal years):** ROA, revenue growth, gross margin, operating margin, SG&A / revenue, gross profit growth, asset turnover (revenue / average assets), SG&A growth, GM change, OM change, OCF / average assets.

**Training sample:** one observation per company. y = FY2025 ROA; x = each ratio averaged over FY2020–FY2024 (growth rates and changes for FY2020 use FY2019 data).

**Steps for each industry group**

1. Compute each ratio every year from the raw columns (`revt, gp, xsga, oiadp, ni, at, oancf`), then average FY2020–2024.
2. Winsorize every variable at the 1st / 99th percentile to limit the effect of outliers.
3. Start with **5-yr avg ROA always in the model**. Try adding each remaining variable, add the one with the lowest p-value if p < 0.05, and repeat until nothing qualifies.
4. The resulting OLS coefficients, covariance matrix and residual variance are stored in `app.py`.

**In the app:** the same ratios are averaged over the company's latest reported fiscal years from Yahoo Finance and plugged into the model for its industry group (detected from Yahoo's industry, can be overridden). Yahoo usually provides about 4 years of annual statements, so some averages cover 3–4 years instead of 5; the app shows how many years each average uses. A manual-entry mode is available as a fallback.

## Results

| Group | Industry | 5-yr avg variables (order entered) | R² | Adj. R² | n | Residual std. error |
|---|---|---|---|---|---|---|
| [2010](training/results/results_2010.md) | Capital Goods | ROA, OM | 0.447 | 0.442 | 275 | 0.084 |
| [2550](training/results/results_2550.md) | Consumer Discretionary Distribution & Retail | ROA, OM change | 0.551 | 0.541 | 94 | 0.073 |
| [3520](training/results/results_3520.md) | Pharmaceuticals, Biotechnology & Life Sciences | ROA, SG&A/rev, OCF/avg assets, OM change, GP growth | 0.876 | 0.869 | 96 | 0.199 |
| [4510](training/results/results_4510.md) | Software & Services | ROA, OM, OM change | 0.725 | 0.718 | 126 | 0.105 |
| [4530](training/results/results_4530.md) | Semiconductors & Semiconductor Equipment | ROA, OM, OM change | 0.630 | 0.615 | 79 | 0.083 |

Click a group to see every forward selection step and the full coefficient table.

## 95% range

The app shows one sentence and a range bar, for example:

> We forecast Caterpillar's FY2026 ROA at **8.1%**, and we're 95% confident it will land between **−8.8% and 25.0%** (± 16.9%).

* Forecast ŷ = intercept + Σ coefficient × input
* **95% range = ŷ ± 2 × s**, where s is the regression's residual standard error (how far its predictions typically miss actual ROA). About 95% of companies land within two standard errors of the prediction. 2 is the rounded t-value; the exact values are 1.97–1.99 for these models.

| Group | s | 95% range |
|---|---|---|
| 2010 | 0.084 | ± 16.9% |
| 2550 | 0.073 | ± 14.6% |
| 3520 | 0.199 | ± 39.9% |
| 4510 | 0.105 | ± 21.0% |
| 4530 | 0.083 | ± 16.7% |

## Run the app

Locally: `pip install -r requirements.txt` then `streamlit run app.py`.

Streamlit Community Cloud: push this repo to GitHub, go to share.streamlit.io, choose **Create app**, pick the repo and branch, and set the main file to `app.py`.

## Re-run training

```
pip install -r training/requirements-training.txt
python training/train_2010.py      # likewise 2550, 3520, 4510, 4530
```

Then copy the new numbers from `training/results/model_<group>.json` into the `MODELS` dictionary in `app.py`.
