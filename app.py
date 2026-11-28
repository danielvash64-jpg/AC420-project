"""
Future ROA Predictor (Streamlit app)
====================================
Predicts next year's return on assets (ROA = net income / average total assets)
for a company in one of five GICS industry groups, using regression models whose
variables were chosen by forward selection (see the training/ folder).

Deploy on Streamlit Community Cloud with this file as the entry point and a
requirements.txt containing: streamlit, pandas, numpy, plotly, openpyxl
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Model coefficients (copied from training/results/model_<group>.json)
# coef     : intercept + slope for each selected variable
# cov      : coefficient covariance matrix (same order as coef)
# sigma2   : residual variance of the regression
# t_crit   : t critical values for each confidence level
# bounds   : 1st / 99th percentile winsorization bounds from the training data
# ---------------------------------------------------------------------------
MODELS = {2010: {'ggroup': 2010,
        'name': 'Capital Goods',
        'variables': ['roa_prev', 'ocf_assets', 'om', 'om_change'],
        'coef': {'const': -0.0023503,
                 'roa_prev': 0.26891168,
                 'ocf_assets': 0.28194079,
                 'om': 0.12090858,
                 'om_change': -0.25593317},
        'cov': [[5.067186e-05, 7.632736e-05, -0.0004031603, 3.921361e-05, 5.360253e-06],
                [7.632736e-05, 0.003247191, -0.001766727, -0.0008676477, 0.001066444],
                [-0.0004031603, -0.001766727, 0.006230919, -0.0006874197, -7.846686e-05],
                [3.921361e-05, -0.0008676477, -0.0006874197, 0.0008630196, -0.0007440131],
                [5.360253e-06, 0.001066444, -7.846686e-05, -0.0007440131, 0.005406131]],
        'sigma2': 0.00630147,
        'df_resid': 269.0,
        't_crit': {'0.80': 1.28471, '0.90': 1.65054, '0.95': 1.96882, '0.99': 2.59423},
        'r2': 0.50921945,
        'adj_r2': 0.5019216,
        'n_obs': 274,
        'bounds': {'roa_prev': [-0.642675, 0.252867],
                   'ocf_assets': [-0.229172, 0.257746],
                   'om': [-1.659148, 0.313681],
                   'om_change': [-0.40017, 0.246764]}},
 2550: {'ggroup': 2550,
        'name': 'Consumer Discretionary Distribution & Retail',
        'variables': ['roa_prev', 'ocf_assets', 'gm_change'],
        'coef': {'const': 0.00331727,
                 'roa_prev': 1.2489402,
                 'ocf_assets': -0.24254921,
                 'gm_change': -0.96796172},
        'cov': [[5.693623e-05, 0.0001188785, -0.0003015454, -0.0008100832],
                [0.0001188785, 0.009725648, -0.00595464, -0.005570628],
                [-0.0003015454, -0.00595464, 0.006209866, 0.004934465],
                [-0.0008100832, -0.005570628, 0.004934465, 0.1187897]],
        'sigma2': 0.0029959,
        'df_resid': 90.0,
        't_crit': {'0.80': 1.29103, '0.90': 1.66196, '0.95': 1.98667, '0.99': 2.63157},
        'r2': 0.74963179,
        'adj_r2': 0.74128618,
        'n_obs': 94,
        'bounds': {'roa_prev': [-0.196128, 0.301869],
                   'ocf_assets': [-0.346863, 0.449511],
                   'gm_change': [-0.031372, 0.058442]}},
 3520: {'ggroup': 3520,
        'name': 'Pharmaceuticals, Biotechnology & Life Sciences',
        'variables': ['roa_prev',
                      'om',
                      'sga_rev',
                      'gm',
                      'om_change',
                      'gp_growth_5y',
                      'ocf_assets',
                      'asset_turn'],
        'coef': {'const': 0.04852062,
                 'roa_prev': 1.17055246,
                 'om': -1.13312726,
                 'sga_rev': -1.05099257,
                 'gm': 0.87697103,
                 'om_change': 0.16777721,
                 'gp_growth_5y': -0.06524404,
                 'ocf_assets': 0.76015718,
                 'asset_turn': -0.13516293},
        'cov': [[0.003925502,
                 -0.001690851,
                 0.003564853,
                 0.003604249,
                 -0.007561699,
                 6.580144e-06,
                 4.176442e-05,
                 0.003898818,
                 -0.001962612],
                [-0.001690851,
                 0.01143041,
                 -0.004833657,
                 -0.00508003,
                 0.007101956,
                 -0.0002825564,
                 0.0004075223,
                 -0.02183316,
                 0.003102445],
                [0.003564853,
                 -0.004833657,
                 0.02200775,
                 0.02161196,
                 -0.02186619,
                 -0.001499022,
                 0.0004491179,
                 0.005730156,
                 -0.003608696],
                [0.003604249,
                 -0.00508003,
                 0.02161196,
                 0.02141457,
                 -0.02190655,
                 -0.001086899,
                 0.0004068171,
                 0.007184856,
                 -0.003663325],
                [-0.007561699,
                 0.007101956,
                 -0.02186619,
                 -0.02190655,
                 0.02899374,
                 0.0006374633,
                 -0.000519668,
                 -0.01356733,
                 0.004087123],
                [6.580144e-06,
                 -0.0002825564,
                 -0.001499022,
                 -0.001086899,
                 0.0006374633,
                 0.0009254534,
                 -0.000114913,
                 0.002253656,
                 -9.756482e-06],
                [4.176442e-05,
                 0.0004075223,
                 0.0004491179,
                 0.0004068171,
                 -0.000519668,
                 -0.000114913,
                 0.000520388,
                 -0.0006427591,
                 -2.379335e-05],
                [0.003898818,
                 -0.02183316,
                 0.005730156,
                 0.007184856,
                 -0.01356733,
                 0.002253656,
                 -0.0006427591,
                 0.05607751,
                 -0.006458862],
                [-0.001962612,
                 0.003102445,
                 -0.003608696,
                 -0.003663325,
                 0.004087123,
                 -9.756482e-06,
                 -2.379335e-05,
                 -0.006458862,
                 0.003366579]],
        'sigma2': 0.02598984,
        'df_resid': 87.0,
        't_crit': {'0.80': 1.29136, '0.90': 1.66256, '0.95': 1.98761, '0.99': 2.63353},
        'r2': 0.921114,
        'adj_r2': 0.91386011,
        'n_obs': 96,
        'bounds': {'roa_prev': [-2.173685, 0.272817],
                   'om': [-49.685041, 0.470455],
                   'sga_rev': [0.087821, 49.436199],
                   'gm': [0.037907, 0.985804],
                   'om_change': [-21.931974, 2.212797],
                   'gp_growth_5y': [-0.188511, 4.516175],
                   'ocf_assets': [-0.955455, 0.345406],
                   'asset_turn': [0.027047, 1.954629]}},
 4510: {'ggroup': 4510,
        'name': 'Software & Services',
        'variables': ['roa_prev', 'ocf_assets', 'gp_growth_5y', 'sga_rev'],
        'coef': {'const': 0.01377059,
                 'roa_prev': 0.64268418,
                 'ocf_assets': 0.32390477,
                 'gp_growth_5y': -0.16046968,
                 'sga_rev': -0.01612919},
        'cov': [[0.0002480851, 0.0003177185, -0.0009539652, -0.0004152649, -5.233235e-05],
                [0.0003177185, 0.005848272, -0.005537504, 0.000534963, 0.0001007366],
                [-0.0009539652, -0.005537504, 0.01003377, -0.0002139951, 0.0001424164],
                [-0.0004152649, 0.000534963, -0.0002139951, 0.003330407, 8.854561e-07],
                [-5.233235e-05, 0.0001007366, 0.0001424164, 8.854561e-07, 5.30942e-05]],
        'sigma2': 0.00900249,
        'df_resid': 121.0,
        't_crit': {'0.80': 1.28859, '0.90': 1.65754, '0.95': 1.97976, '0.99': 2.61707},
        'r2': 0.77629018,
        'adj_r2': 0.76889481,
        'n_obs': 126,
        'bounds': {'roa_prev': [-0.723252, 0.373449],
                   'ocf_assets': [-0.421417, 0.405351],
                   'gp_growth_5y': [-0.147755, 0.808232],
                   'sga_rev': [0.093764, 10.882958]}},
 4530: {'ggroup': 4530,
        'name': 'Semiconductors & Semiconductor Equipment',
        'variables': ['roa_prev', 'om', 'ocf_assets'],
        'coef': {'const': -0.02300403, 'roa_prev': -0.07029008, 'om': 0.22128564, 'ocf_assets': 0.50530094},
        'cov': [[0.0001689126, 0.0001319616, 0.0003429304, -0.001254168],
                [0.0001319616, 0.004834481, -0.0007482204, -0.003285705],
                [0.0003429304, -0.0007482204, 0.002816625, -0.004707518],
                [-0.001254168, -0.003285705, -0.004707518, 0.01728919]],
        'sigma2': 0.00580254,
        'df_resid': 75.0,
        't_crit': {'0.80': 1.29294, '0.90': 1.66543, '0.95': 1.9921, '0.99': 2.64298},
        'r2': 0.69033476,
        'adj_r2': 0.67794816,
        'n_obs': 79,
        'bounds': {'roa_prev': [-0.551187, 0.641427],
                   'om': [-0.966508, 0.619487],
                   'ocf_assets': [-0.251384, 0.452757]}}}

# Example inputs (FY2020-2025, USD millions) so the app has something to show
EXAMPLES = {2010: {'ticker': 'CAT',
        'rows': [[2020, 41748.0, 14384.0, 6057.0, 5894.0, 2998.0, 78324.0, 6327.0],
                 [2021, 50971.0, 17114.0, 7051.0, 7711.0, 6489.0, 82793.0, 7198.0],
                 [2022, 59427.0, 19795.0, 7465.0, 10109.0, 6705.0, 81943.0, 7766.0],
                 [2023, 67060.0, 25837.0, 8479.0, 15211.0, 10335.0, 87476.0, 12885.0],
                 [2024, 64809.0, 26165.0, 8774.0, 15232.0, 10792.0, 87764.0, 12035.0],
                 [2025, 67589.0, 24691.0, 9133.0, 13296.0, 8884.0, 98585.0, 11739.0]]},
 2550: {'ticker': 'HD',
        'rows': [[2020, 132110.0, 45150.0, 22099.0, 20626.0, 12866.0, 70581.0, 18839.0],
                 [2021, 151157.0, 51096.0, 25144.0, 23302.0, 16433.0, 71876.0, 16571.0],
                 [2022, 157403.0, 53079.0, 26284.0, 24039.0, 17105.0, 76445.0, 14615.0],
                 [2023, 152669.0, 51307.0, 26598.0, 21689.0, 15143.0, 76530.0, 21172.0],
                 [2024, 159514.0, 53982.0, 28748.0, 21526.0, 14806.0, 96119.0, 19810.0],
                 [2025, 164683.0, 55651.0, 30702.0, 20890.0, 14156.0, 105095.0, 16325.0]]},
 3520: {'ticker': 'PFE',
        'rows': [[2020, 41908.0, 34669.0, 20605.0, 9287.0, 9616.0, 154229.0, 14403.0],
                 [2021, 81288.0, 52065.0, 23841.0, 23033.0, 21979.0, 181476.0, 32580.0],
                 [2022, 100330.0, 67647.0, 24534.0, 38049.0, 31372.0, 197205.0, 29267.0],
                 [2023, 61996.0, 38984.0, 25160.0, 7534.0, 2119.0, 226501.0, 8700.0],
                 [2024, 62856.0, 46866.0, 25373.0, 14480.0, 8031.0, 213396.0, 12744.0],
                 [2025, 62579.0, 48416.0, 23925.0, 17899.0, 7771.0, 208160.0, 11704.0]]},
 4510: {'ticker': 'MSFT',
        'rows': [[2020, 143015.0, 109237.0, 43978.0, 52959.0, 44281.0, 301311.0, 60675.0],
                 [2021, 168088.0, 126756.0, 45940.0, 69916.0, 61271.0, 333779.0, 76740.0],
                 [2022, 198270.0, 150220.0, 52111.0, 83509.0, 72738.0, 364840.0, 89035.0],
                 [2023, 211915.0, 159704.0, 56510.0, 89694.0, 72361.0, 411976.0, 87582.0],
                 [2024, 245122.0, 191008.0, 61575.0, 109433.0, 88136.0, 512163.0, 118548.0],
                 [2025, 281724.0, 221893.0, 65365.0, 128528.0, 101832.0, 619003.0, 136162.0]]},
 4530: {'ticker': 'TXN',
        'rows': [[2020, 14461.0, 10063.0, 3153.0, 5918.0, 5595.0, 19351.0, 6139.0],
                 [2021, 18344.0, 13188.0, 3324.0, 8910.0, 7769.0, 24676.0, 8756.0],
                 [2022, 20028.0, 14750.0, 3631.0, 10140.0, 8749.0, 27207.0, 8720.0],
                 [2023, 17519.0, 12257.0, 3688.0, 7331.0, 6510.0, 32348.0, 6420.0],
                 [2024, 15641.0, 10674.0, 3753.0, 5341.0, 4799.0, 35509.0, 6318.0],
                 [2025, 17682.0, 12082.0, 3943.0, 6140.0, 5001.0, 34585.0, 7153.0]]}}

VARIABLE_LABELS = {
    "roa_prev": "Previous year ROA",
    "rev_growth_5y": "5-yr avg revenue growth",
    "gm": "Gross margin (GM)",
    "om": "Operating margin (OM)",
    "sga_rev": "SG&A / revenue",
    "gp_growth_5y": "5-yr avg gross profit growth",
    "asset_turn": "Asset turnover (revenue / avg assets)",
    "sga_growth_5y": "5-yr avg SG&A growth",
    "gm_change": "GM change",
    "om_change": "OM change",
    "ocf_assets": "OCF / avg assets",
}

INPUT_COLUMNS = {
    "fyear": "Fiscal year",
    "revt": "Revenue",
    "gp": "Gross profit",
    "xsga": "SG&A",
    "oiadp": "Operating income",
    "ni": "Net income",
    "at": "Total assets",
    "oancf": "Operating cash flow",
}

BLUE = "#257de0"
ORANGE = "#f47230"
GRAY = "#8a8a8a"


# ---------------------------------------------------------------------------
# Variable construction (same definitions as training/roa_model.py)
# ---------------------------------------------------------------------------
def _div(a, b):
    return a / b if (b is not None and b != 0 and not pd.isna(b) and not pd.isna(a)) else np.nan


def _avg_growth(values):
    """Average of the last five year-over-year growth rates (needs >= 3 valid)."""
    v = list(values)[-6:]
    rates = [(v[i] - v[i - 1]) / v[i - 1] for i in range(1, len(v))
             if not pd.isna(v[i]) and not pd.isna(v[i - 1]) and v[i - 1] > 0]
    return float(np.mean(rates)) if len(rates) >= 3 else np.nan


def compute_variables(fin: pd.DataFrame) -> dict:
    """fin: one row per fiscal year, sorted oldest -> newest, Compustat column names."""
    f = fin.sort_values("fyear").reset_index(drop=True)
    cur, prev = f.iloc[-1], f.iloc[-2]
    avg_at = (cur["at"] + prev["at"]) / 2
    gm, gm_prev = _div(cur["gp"], cur["revt"]), _div(prev["gp"], prev["revt"])
    om, om_prev = _div(cur["oiadp"], cur["revt"]), _div(prev["oiadp"], prev["revt"])
    return {
        "roa_prev": _div(cur["ni"], avg_at),
        "rev_growth_5y": _avg_growth(f["revt"]),
        "gm": gm,
        "om": om,
        "sga_rev": _div(cur["xsga"], cur["revt"]),
        "gp_growth_5y": _avg_growth(f["gp"]),
        "asset_turn": _div(cur["revt"], avg_at),
        "sga_growth_5y": _avg_growth(f["xsga"]),
        "gm_change": gm - gm_prev,
        "om_change": om - om_prev,
        "ocf_assets": _div(cur["oancf"], avg_at),
    }


def roa_history(fin: pd.DataFrame) -> pd.DataFrame:
    f = fin.sort_values("fyear").reset_index(drop=True)
    avg_at = (f["at"] + f["at"].shift(1)) / 2
    return pd.DataFrame({"fyear": f["fyear"], "roa": f["ni"] / avg_at}).dropna()


def predict(model: dict, values: dict, level: float):
    names = model["variables"]
    x = np.array([1.0] + [values[n] for n in names])
    b = np.array([model["coef"]["const"]] + [model["coef"][n] for n in names])
    V = np.array(model["cov"])
    yhat = float(x @ b)
    se_mean = math.sqrt(float(x @ V @ x))
    se_pred = math.sqrt(model["sigma2"] + se_mean ** 2)
    t = model["t_crit"][f"{level:.2f}"]
    return {
        "yhat": yhat,
        "se_mean": se_mean,
        "se_pred": se_pred,
        "pi": (yhat - t * se_pred, yhat + t * se_pred),
        "ci": (yhat - t * se_mean, yhat + t * se_mean),
        "contrib": {n: model["coef"][n] * values[n] for n in names},
    }


def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Future ROA Predictor", layout="wide")
st.title("Future ROA Predictor")
st.caption("Predicts next year's ROA (net income / average total assets) with a regression model "
           "built by forward selection, separately for each GICS industry group.")

with st.sidebar:
    st.header("Settings")
    group_options = list(MODELS.keys())
    source = st.radio("Input method", ["Enter financials", "Upload spreadsheet"])
    level = st.select_slider("Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda v: f"{v:.0%}")

# ---------------- Inputs ----------------
fin, company, group = None, "", None

if source == "Enter financials":
    with st.sidebar:
        group = st.selectbox("GICS industry group", group_options,
                             format_func=lambda g: f"{g} · {MODELS[g]['name']}")
    ex = EXAMPLES[group]
    st.subheader("1. Company financials (USD millions)")
    st.write(f"Enter at least the last 6 fiscal years, oldest first. The table starts with "
             f"**{ex['ticker']}** as an example; replace it with your company's numbers.")
    company = st.text_input("Company name or ticker", value=ex["ticker"], key=f"name_{group}")
    default = pd.DataFrame(ex["rows"], columns=list(INPUT_COLUMNS)).rename(columns=INPUT_COLUMNS)
    default["Fiscal year"] = default["Fiscal year"].astype(int)
    edited = st.data_editor(default, num_rows="dynamic", width="stretch", hide_index=True,
                            key=f"editor_{group}")
    fin = edited.rename(columns={v: k for k, v in INPUT_COLUMNS.items()}).dropna(subset=["fyear"])
else:
    st.subheader("1. Upload company data")
    st.write("Upload an .xlsx or .csv in the same layout as the Compustat export "
             "(columns: tic, fyear, ggroup, revt, gp, xsga, oiadp, ni, at, oancf).")
    up = st.file_uploader("Spreadsheet", type=["xlsx", "csv"])
    if up is None:
        st.info("Upload a file to continue, or switch to **Enter financials** in the sidebar.")
        st.stop()
    data = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
    data.columns = [c.strip().lower() for c in data.columns]
    missing = [c for c in ["tic", "fyear", "ggroup"] + list(INPUT_COLUMNS)[1:] if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()
    data = data[data["ggroup"].isin(group_options)]
    if data.empty:
        st.error("No rows in industry groups " + ", ".join(map(str, group_options)) + ".")
        st.stop()
    c1, c2 = st.columns(2)
    company = c1.selectbox("Company (ticker)", sorted(data["tic"].astype(str).unique()))
    firm = data[data["tic"].astype(str) == company].sort_values("fyear")
    years = firm["fyear"].astype(int).tolist()
    base = c2.selectbox("Most recent fiscal year to use", years[::-1])
    group = int(firm["ggroup"].iloc[0])
    fin = firm[firm["fyear"] <= base][list(INPUT_COLUMNS)].tail(6)
    st.caption(f"Industry group detected: {group} · {MODELS[group]['name']}")
    st.dataframe(fin.rename(columns=INPUT_COLUMNS), hide_index=True, width="stretch")

fin = fin.apply(pd.to_numeric, errors="coerce").sort_values("fyear")
if len(fin) < 2:
    st.warning("Enter at least two fiscal years.")
    st.stop()
if len(fin) < 6:
    st.warning("Fewer than 6 years entered: 5-year average growth rates use only the years available "
               "(at least 4 years are needed).")

model = MODELS[group]
base_year = int(fin["fyear"].iloc[-1])
raw_vals = compute_variables(fin)

missing_vars = [VARIABLE_LABELS[v] for v in model["variables"] if pd.isna(raw_vals[v])]
if missing_vars:
    st.error("Can't compute: " + ", ".join(missing_vars) + ". Check for blank, zero or negative values.")
    st.stop()

# Clip inputs to the range seen in training (same winsorization as the model)
vals, clipped = dict(raw_vals), []
for v in model["variables"]:
    lo, hi = model["bounds"][v]
    if raw_vals[v] < lo or raw_vals[v] > hi:
        clipped.append(VARIABLE_LABELS[v])
    vals[v] = min(max(raw_vals[v], lo), hi)

res = predict(model, vals, level)
lo_pi, hi_pi = res["pi"]
lo_ci, hi_ci = res["ci"]
p_positive = normal_cdf(res["yhat"] / res["se_pred"])

# ---------------- Headline ----------------
st.subheader(f"2. Predicted ROA for FY{base_year + 1}" + (f" · {company}" if company else ""))
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Predicted ROA", f"{res['yhat']:.1%}",
          delta=f"{res['yhat'] - raw_vals['roa_prev']:+.1%} vs FY{base_year}", delta_color="normal")
m2.metric(f"± margin ({level:.0%})", f"± {(hi_pi - lo_pi) / 2:.1%}",
          help="Half-width of the prediction interval: the range where this company's actual ROA should "
               "fall with the chosen confidence. Includes model uncertainty and company-specific noise.")
m3.metric(f"Low end ({level:.0%})", f"{lo_pi:.1%}")
m4.metric(f"High end ({level:.0%})", f"{hi_pi:.1%}")
m5.metric("Chance ROA is positive", f"{p_positive:.0%}")
st.caption(f"{level:.0%} confidence interval for the average ROA of companies with these same ratios: "
           f"{lo_ci:.1%} to {hi_ci:.1%} (narrower, because it excludes company-specific noise).")
if clipped:
    st.warning("Outside the training range, so capped at the 1st/99th percentile: " + ", ".join(clipped))

# ---------------- Charts ----------------
hist = roa_history(fin)
left, right = st.columns([3, 2])

with left:
    fig = go.Figure()
    nx = base_year + 1
    # nested prediction-interval bands, widest first
    t_levels = sorted(model["t_crit"].keys(), reverse=True)
    for i, key in enumerate(t_levels):
        t = model["t_crit"][key]
        lo, hi = res["yhat"] - t * res["se_pred"], res["yhat"] + t * res["se_pred"]
        fig.add_trace(go.Scatter(
            x=[base_year, nx, nx, base_year], y=[hist["roa"].iloc[-1], hi, lo, hist["roa"].iloc[-1]],
            fill="toself", mode="lines", line=dict(width=0), fillcolor=BLUE,
            opacity=0.10 + 0.08 * i, name=f"{float(key):.0%} prediction interval",
            hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["fyear"], y=hist["roa"], mode="lines+markers", name="Actual ROA",
                             line=dict(color="#333333", width=2), marker=dict(size=8),
                             hovertemplate="FY%{x}: %{y:.1%}<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(
        x=[nx], y=[res["yhat"]], mode="markers", name=f"Prediction (±{level:.0%} CI of mean)",
        marker=dict(size=12, color=BLUE, line=dict(color="white", width=2)),
        error_y=dict(type="data", symmetric=False, array=[hi_ci - res["yhat"]],
                     arrayminus=[res["yhat"] - lo_ci], color=BLUE, thickness=2, width=8),
        hovertemplate=(f"FY{nx} predicted: %{{y:.1%}}<br>{level:.0%} PI: {lo_pi:.1%} to {hi_pi:.1%}"
                       f"<br>{level:.0%} CI: {lo_ci:.1%} to {hi_ci:.1%}<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1, dash="dot"))
    fig.update_layout(title="ROA history and forecast fan", yaxis_tickformat=".0%",
                      xaxis=dict(dtick=1, title="Fiscal year"), yaxis_title="ROA",
                      legend=dict(orientation="h", y=-0.25), height=440, margin=dict(t=50, b=10),
                      hovermode="closest", plot_bgcolor="white")
    fig.update_yaxes(gridcolor="#eeeeee")
    st.plotly_chart(fig, width="stretch")

with right:
    xs = np.linspace(res["yhat"] - 4 * res["se_pred"], res["yhat"] + 4 * res["se_pred"], 400)
    dens = np.exp(-0.5 * ((xs - res["yhat"]) / res["se_pred"]) ** 2) / (res["se_pred"] * math.sqrt(2 * math.pi))
    inside = (xs >= lo_pi) & (xs <= hi_pi)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=dens, mode="lines", line=dict(color=BLUE, width=2),
                              name="Likelihood", hovertemplate="ROA %{x:.1%}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=xs[inside], y=dens[inside], fill="tozeroy", mode="none",
                              fillcolor="rgba(37,125,224,0.25)", name=f"{level:.0%} interval",
                              hoverinfo="skip"))
    fig2.add_vline(x=res["yhat"], line=dict(color=BLUE, width=2))
    fig2.add_vline(x=0, line=dict(color=GRAY, width=1, dash="dot"))
    fig2.add_annotation(x=res["yhat"], y=dens.max(), text=f"{res['yhat']:.1%}", showarrow=False, yshift=12)
    fig2.update_layout(title="How confident is the prediction?", xaxis_tickformat=".0%",
                       xaxis_title=f"FY{nx} ROA", yaxis=dict(visible=False), showlegend=False,
                       height=440, margin=dict(t=50, b=10), plot_bgcolor="white")
    st.plotly_chart(fig2, width="stretch")

# Contribution of each variable
contrib = pd.DataFrame({"Variable": [VARIABLE_LABELS[v] for v in res["contrib"]],
                        "Contribution": list(res["contrib"].values())})
fig3 = go.Figure(go.Bar(
    x=contrib["Contribution"], y=contrib["Variable"], orientation="h",
    marker_color=[BLUE if c >= 0 else ORANGE for c in contrib["Contribution"]],
    hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
fig3.update_layout(title=f"What drives the prediction (intercept {model['coef']['const']:+.2%} not shown)",
                   xaxis_tickformat="+.1%", height=120 + 40 * len(contrib), margin=dict(t=50, b=10),
                   plot_bgcolor="white")
fig3.update_xaxes(gridcolor="#eeeeee", zerolinecolor=GRAY)
st.plotly_chart(fig3, width="stretch")

# ---------------- Details ----------------
st.subheader("3. Inputs and model details")
detail = pd.DataFrame({
    "Variable": [VARIABLE_LABELS[v] for v in model["variables"]],
    "Your value": [raw_vals[v] for v in model["variables"]],
    "Value used": [vals[v] for v in model["variables"]],
    "Coefficient": [model["coef"][v] for v in model["variables"]],
    "Contribution": [res["contrib"][v] for v in model["variables"]],
})
st.dataframe(detail.style.format({"Your value": "{:.4f}", "Value used": "{:.4f}",
                                  "Coefficient": "{:.4f}", "Contribution": "{:+.4f}"}),
             hide_index=True, width="stretch")

with st.expander("How this works"):
    eq = f"{model['coef']['const']:.4f} " + " ".join(
        f"{'+' if model['coef'][v] >= 0 else '−'} {abs(model['coef'][v]):.4f} × {VARIABLE_LABELS[v]}"
        for v in model["variables"])
    st.markdown(f"""
**Model for GICS {group} ({model['name']})**

Future ROA = {eq}

* Previous year ROA is always in the model; the other variables were added by forward selection
  (entry rule p < 0.05) from the remaining 10 candidates.
* Trained on the most recent year: FY2024 ratios → 2025 ROA (one observation per company).
  Enter FY2025 financials (and the 5 years before) to predict 2026.
* R² = {model['r2']:.3f}, adjusted R² = {model['adj_r2']:.3f}, n = {model['n_obs']} firm-years,
  residual standard error = {math.sqrt(model['sigma2']):.4f}.
* **Prediction interval** = ŷ ± t × √(s² + x′Vx), where s² is the residual variance and V is the
  coefficient covariance matrix. **Confidence interval** (average company) = ŷ ± t × √(x′Vx).
* Assumes OLS conditions hold (roughly normal, constant-variance errors). Firm-years from the same
  company are treated as independent, so true uncertainty may be somewhat larger.

Other variables computed from your inputs (not in this industry's model):
""")
    others = {VARIABLE_LABELS[k]: v for k, v in raw_vals.items() if k not in model["variables"]}
    st.dataframe(pd.DataFrame({"Variable": list(others), "Value": list(others.values())})
                 .style.format({"Value": "{:.4f}"}), hide_index=True, width="stretch")
