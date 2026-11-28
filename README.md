# Future ROA Predictor

A Streamlit app that predicts a company's next-year return on assets (ROA) using regression models built with **forward selection**, one per GICS industry group (2010, 2550, 3520, 4510, 4530). Enter a company's FY2025 financials (plus the five years before) to predict its 2026 ROA, with a ± prediction interval and a confidence interval.

## Method

**Target:** ROA in year t+1, where ROA = net income / average total assets.

**Candidate variables (year t):** previous year ROA, 5-yr average revenue growth, gross margin, operating margin, SG&A / revenue, 5-yr average gross profit growth, asset turnover (revenue / average assets), 5-yr average SG&A growth, GM change, OM change, OCF / average assets.

**Training sample:** the most recent year with a known outcome: FY2024 ratios → 2025 ROA, one observation per company. The 5-year growth averages reach back to FY2019.

**Steps for each industry group**

1. Build all variables from the raw columns (`revt, gp, xsga, oiadp, ni, at, oancf`).
2. Winsorize every variable at the 1st / 99th percentile to limit the effect of outliers.
3. Start with **previous year ROA always in the model**. Try adding each remaining variable, add the one with the lowest p-value if p < 0.05, and repeat until nothing qualifies.
4. The resulting OLS coefficients, covariance matrix and residual variance are stored in `app.py`.

## Results

| Group | Industry | Variables (order entered) | R² | Adj. R² | n | Residual std. error |
|---|---|---|---|---|---|---|
| [2010](training/results/results_2010.md) | Capital Goods | Prev. year ROA, OCF/avg assets, OM, OM change | 0.509 | 0.502 | 274 | 0.079 |
| [2550](training/results/results_2550.md) | Consumer Discretionary Distribution & Retail | Prev. year ROA, OCF/avg assets, GM change | 0.750 | 0.741 | 94 | 0.055 |
| [3520](training/results/results_3520.md) | Pharmaceuticals, Biotechnology & Life Sciences | Prev. year ROA, OM, SG&A/rev, GM, OM change, 5-yr GP growth, OCF/avg assets, Asset turnover | 0.921 | 0.914 | 96 | 0.161 |
| [4510](training/results/results_4510.md) | Software & Services | Prev. year ROA, OCF/avg assets, 5-yr GP growth, SG&A/rev | 0.776 | 0.769 | 126 | 0.095 |
| [4530](training/results/results_4530.md) | Semiconductors & Semiconductor Equipment | Prev. year ROA, OM, OCF/avg assets | 0.690 | 0.678 | 79 | 0.076 |

Click a group to see every forward selection step and the full coefficient table.

## Prediction and confidence intervals

For inputs x, coefficients b, coefficient covariance V and residual variance s²:

* Prediction ŷ = x′b
* **Prediction interval** (where this company's ROA should land) = ŷ ± t · √(s² + x′Vx)
* **Confidence interval** (average ROA of companies with these ratios) = ŷ ± t · √(x′Vx)

## Run the app

Locally: `pip install -r requirements.txt` then `streamlit run app.py`.

Streamlit Community Cloud: push this repo to GitHub, go to share.streamlit.io, choose **Create app**, pick the repo and branch, and set the main file to `app.py`.

## Re-run training

```
pip install -r training/requirements-training.txt
python training/train_2010.py      # likewise 2550, 3520, 4510, 4530
```

Then copy the new numbers from `training/results/model_<group>.json` into the `MODELS` dictionary in `app.py`.
