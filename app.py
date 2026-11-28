"""
Future ROA Predictor (Streamlit app)
====================================
Enter a stock ticker. The app pulls the company's most recently reported annual
financial statements from Yahoo Finance (yfinance), averages each ratio over the
latest (up to) 5 fiscal years, and applies the industry's regression model to
forecast ROA (net income / average total assets) for the next fiscal year,
with a 95% range of forecast ± 2 × s.

The models were built by forward selection (see the training/ folder):
y = FY2025 ROA, x = 5-year averages of the ratios over FY2020-2024.

Deploy on Streamlit Community Cloud with this file as the entry point and a
requirements.txt containing: streamlit, pandas, numpy, plotly, yfinance
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Final regression models (copied from training/results/model_<group>.json)
# coef     : intercept + slope for each selected variable
# s        : residual standard error of the regression (typical miss)
# bounds   : 1st / 99th percentile winsorization bounds from the training data
# ---------------------------------------------------------------------------
MODELS = {2010: {'ggroup': 2010,
        'name': 'Capital Goods',
        'variables': ['roa', 'om'],
        'coef': {'const': 0.02073764, 'roa': 0.35015687, 'om': 0.1186488},
        's': 0.084409,
        'r2': 0.446541,
        'adj_r2': 0.442471,
        'n_obs': 275,
        'bounds': {'roa': [-0.546129, 0.239847], 'om': [-2.282879, 0.276537]}},
 2550: {'ggroup': 2550,
        'name': 'Consumer Discretionary Distribution & Retail',
        'variables': ['roa', 'om_change'],
        'coef': {'const': -0.02002861, 'roa': 0.80966756, 'om_change': 2.86645057},
        's': 0.072895,
        'r2': 0.550998,
        'adj_r2': 0.54113,
        'n_obs': 94,
        'bounds': {'roa': [-0.106713, 0.319114], 'om_change': [-0.028475, 0.036053]}},
 3520: {'ggroup': 3520,
        'name': 'Pharmaceuticals, Biotechnology & Life Sciences',
        'variables': ['roa', 'sga_rev', 'ocf_assets', 'om_change', 'gp_growth'],
        'coef': {'const': -0.09811401,
                 'roa': 0.34940555,
                 'sga_rev': 0.05285226,
                 'ocf_assets': 1.18574376,
                 'om_change': 0.07101903,
                 'gp_growth': -0.09034091},
        's': 0.199309,
        'r2': 0.876368,
        'adj_r2': 0.869499,
        'n_obs': 96,
        'bounds': {'roa': [-5.322966, 0.244591],
                   'sga_rev': [0.102005, 54.487425],
                   'ocf_assets': [-1.595405, 0.355334],
                   'om_change': [-21.256326, 3.832661],
                   'gp_growth': [-0.188511, 3.139548]}},
 4510: {'ggroup': 4510,
        'name': 'Software & Services',
        'variables': ['roa', 'om', 'om_change'],
        'coef': {'const': 0.03055778, 'roa': 0.89845026, 'om': 0.05472551, 'om_change': 0.23418598},
        's': 0.104857,
        'r2': 0.724518,
        'adj_r2': 0.717744,
        'n_obs': 126,
        'bounds': {'roa': [-0.660508, 0.258358],
                   'om': [-8.328956, 0.418772],
                   'om_change': [-0.657605, 0.402116]}},
 4530: {'ggroup': 4530,
        'name': 'Semiconductors & Semiconductor Equipment',
        'variables': ['roa', 'om', 'om_change'],
        'coef': {'const': -0.0213532, 'roa': 0.37843411, 'om': 0.2662465, 'om_change': 0.57105381},
        's': 0.083282,
        'r2': 0.629848,
        'adj_r2': 0.615042,
        'n_obs': 79,
        'bounds': {'roa': [-0.284971, 0.295745],
                   'om': [-0.624936, 0.481827],
                   'om_change': [-0.161598, 0.138538]}}}

# Example inputs for manual entry (FY2020-2025, USD millions)
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
    "roa": "5-yr avg ROA",
    "rev_growth": "5-yr avg revenue growth",
    "gm": "5-yr avg gross margin",
    "om": "5-yr avg operating margin",
    "sga_rev": "5-yr avg SG&A / revenue",
    "gp_growth": "5-yr avg gross profit growth",
    "asset_turn": "5-yr avg asset turnover",
    "sga_growth": "5-yr avg SG&A growth",
    "gm_change": "5-yr avg GM change",
    "om_change": "5-yr avg OM change",
    "ocf_assets": "5-yr avg OCF / avg assets",
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

# Yahoo Finance "industry" -> GICS industry group (best match; the user can override)
YAHOO_TO_GICS = {
    # 2010 Capital Goods
    "Aerospace & Defense": 2010, "Farm & Heavy Construction Machinery": 2010,
    "Specialty Industrial Machinery": 2010, "Electrical Equipment & Parts": 2010,
    "Building Products & Equipment": 2010, "Engineering & Construction": 2010,
    "Industrial Distribution": 2010, "Conglomerates": 2010, "Tools & Accessories": 2010,
    "Metal Fabrication": 2010, "Pollution & Treatment Controls": 2010,
    "Infrastructure Operations": 2010,
    # 2550 Consumer Discretionary Distribution & Retail
    "Internet Retail": 2550, "Specialty Retail": 2550, "Home Improvement Retail": 2550,
    "Department Stores": 2550, "Apparel Retail": 2550, "Auto & Truck Dealerships": 2550,
    # 3520 Pharmaceuticals, Biotechnology & Life Sciences
    "Drug Manufacturers - General": 3520, "Drug Manufacturers - Specialty & Generic": 3520,
    "Biotechnology": 3520, "Diagnostics & Research": 3520,
    # 4510 Software & Services
    "Software - Application": 4510, "Software - Infrastructure": 4510,
    "Information Technology Services": 4510,
    # 4530 Semiconductors & Semiconductor Equipment
    "Semiconductors": 4530, "Semiconductor Equipment & Materials": 4530,
}

# Yahoo statement line items (first one found is used)
YF_FIELDS = {
    "revt": ("income", ["Total Revenue", "Operating Revenue"]),
    "gp": ("income", ["Gross Profit"]),
    "xsga": ("income", ["Selling General And Administration"]),
    "oiadp": ("income", ["Operating Income", "Total Operating Income As Reported", "EBIT"]),
    "ni": ("income", ["Net Income", "Net Income Common Stockholders",
                      "Net Income From Continuing Operation Net Minority Interest"]),
    "at": ("balance", ["Total Assets"]),
    "oancf": ("cash", ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
}

BLUE = "#257de0"
ORANGE = "#f47230"
GRAY = "#8a8a8a"


# ---------------------------------------------------------------------------
# Data retrieval from Yahoo Finance
# ---------------------------------------------------------------------------
def _pick(df: pd.DataFrame, names):
    for n in names:
        if df is not None and n in df.index:
            return df.loc[n]
    return None


def _fiscal_year(ts) -> int:
    """Compustat convention: fiscal years ending Jan-May belong to the prior year."""
    ts = pd.Timestamp(ts)
    return ts.year - 1 if ts.month <= 5 else ts.year


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_yahoo(ticker: str):
    t = yf.Ticker(ticker)
    stmts = {"income": t.income_stmt, "balance": t.balance_sheet, "cash": t.cashflow}
    if stmts["income"] is None or stmts["income"].empty:
        return None, {}, []
    data, notes = {}, []
    for col, (which, names) in YF_FIELDS.items():
        s = _pick(stmts[which], names)
        if s is None and col == "gp":
            rev, cost = _pick(stmts["income"], ["Total Revenue"]), _pick(stmts["income"], ["Cost Of Revenue"])
            if rev is not None and cost is not None:
                s = rev - cost
                notes.append("Gross profit = revenue − cost of revenue.")
        if s is None and col == "xsga":
            parts = [_pick(stmts["income"], [n]) for n in
                     ["Selling And Marketing Expense", "General And Administrative Expense"]]
            parts = [p for p in parts if p is not None]
            if parts:
                s = sum(parts)
                notes.append("SG&A = selling & marketing + general & administrative.")
        data[col] = s
    dates = sorted(set().union(*[set(s.index) for s in data.values() if s is not None]))
    rows = []
    for d in dates:
        row = {"fyear": _fiscal_year(d)}
        for col, s in data.items():
            v = s.get(d, np.nan) if s is not None else np.nan
            row[col] = float(v) / 1e6 if pd.notna(v) else np.nan   # millions
        rows.append(row)
    fin = (pd.DataFrame(rows).groupby("fyear", as_index=False).last()
           .sort_values("fyear").reset_index(drop=True))
    # drop years that are mostly empty (Yahoo sometimes pads an old column)
    fin = fin[fin[list(YF_FIELDS)].notna().sum(axis=1) >= 4].reset_index(drop=True)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    meta = {k: info.get(k) for k in ["longName", "sector", "industry", "financialCurrency"]}
    return fin, meta, notes


# ---------------------------------------------------------------------------
# Variable construction (same definitions as training/roa_model.py)
# ---------------------------------------------------------------------------
def yearly_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    f = fin.sort_values("fyear").reset_index(drop=True).astype(float)

    def div(a, b):
        return a / b.where(b != 0)

    def growth(s):
        prev = s.shift(1)
        return (s - prev) / prev.where(prev > 0)

    out = pd.DataFrame({"fyear": f["fyear"].astype(int)})
    avg_at = (f["at"] + f["at"].shift(1)) / 2
    out["roa"] = div(f["ni"], avg_at)
    out["gm"] = div(f["gp"], f["revt"])
    out["om"] = div(f["oiadp"], f["revt"])
    out["sga_rev"] = div(f["xsga"], f["revt"])
    out["asset_turn"] = div(f["revt"], avg_at)
    out["ocf_assets"] = div(f["oancf"], avg_at)
    out["gm_change"] = out["gm"] - out["gm"].shift(1)
    out["om_change"] = out["om"] - out["om"].shift(1)
    out["rev_growth"] = growth(f["revt"])
    out["gp_growth"] = growth(f["gp"])
    out["sga_growth"] = growth(f["xsga"])
    return out.replace([np.inf, -np.inf], np.nan)


def five_year_averages(yr: pd.DataFrame, min_years: int = 2):
    """Average each ratio over the latest 5 fiscal years (fewer if that's all there is)."""
    window = yr.tail(5)
    avgs, counts = {}, {}
    for v in VARIABLE_LABELS:
        s = window[v].dropna()
        counts[v] = len(s)
        avgs[v] = float(s.mean()) if len(s) >= min_years else np.nan
    return avgs, counts


def predict(model: dict, values: dict):
    """Forecast = intercept + sum(coefficient x input). 95% range = forecast ± 2 x s."""
    names = model["variables"]
    yhat = model["coef"]["const"] + sum(model["coef"][n] * values[n] for n in names)
    margin = 2 * model["s"]
    return {
        "yhat": yhat,
        "margin": margin,
        "low": yhat - margin,
        "high": yhat + margin,
        "contrib": {n: model["coef"][n] * values[n] for n in names},
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Future ROA Predictor", layout="wide")
st.title("Future ROA Predictor")
st.caption("Enter a ticker to pull its latest annual financial statements from Yahoo Finance. The app "
           "averages each ratio over the most recent (up to) 5 fiscal years and forecasts next year's ROA "
           "(net income / average total assets) with the regression model for its GICS industry group.")

group_options = list(MODELS.keys())


def fmt_group(g):
    return f"{g} · {MODELS[g]['name']}"


with st.sidebar:
    st.header("Settings")
    source = st.radio("Data source", ["Yahoo Finance (ticker)", "Enter financials manually"])

# ---------------- Inputs ----------------
fin, company, group = None, "", None

if source == "Yahoo Finance (ticker)":
    st.subheader("1. Company")
    c1, c2 = st.columns([1, 2])
    ticker = c1.text_input("Stock ticker", value="CAT").strip().upper()
    if not ticker:
        st.stop()
    with st.spinner(f"Retrieving {ticker} from Yahoo Finance..."):
        try:
            fin, meta, notes = fetch_yahoo(ticker)
        except Exception as e:
            fin, meta, notes = None, {}, []
            st.error(f"Yahoo Finance request failed: {e}")
    if fin is None or fin.empty:
        st.error(f"No annual financial statements found for {ticker}. Check the ticker, try again in a "
                 f"minute (Yahoo sometimes rate-limits), or switch to manual entry in the sidebar.")
        st.stop()
    company = meta.get("longName") or ticker
    yahoo_ind = meta.get("industry")
    guess = YAHOO_TO_GICS.get(yahoo_ind)
    with c2:
        st.markdown(f"**{company}** · Yahoo industry: {yahoo_ind or 'n/a'} · "
                    f"reporting currency: {meta.get('financialCurrency') or 'n/a'}")
        group = st.selectbox("GICS industry group (model to use)", group_options,
                             index=group_options.index(guess) if guess in group_options else 0,
                             format_func=fmt_group, key=f"grp_{ticker}")
    if guess not in group_options:
        st.warning(f"Yahoo lists {company} as \"{yahoo_ind}\", which isn't clearly in one of the five "
                   f"industry groups the models cover. Pick the closest group above; the forecast may be "
                   f"less reliable.")
    st.markdown("**Most recently reported annual financials (millions, from Yahoo Finance).** "
                "You can correct any cell; blank cells are skipped in the averages.")
    fin = st.data_editor(fin.rename(columns=INPUT_COLUMNS), hide_index=True, width="stretch",
                         disabled=["Fiscal year"], key=f"yf_{ticker}")
    fin = fin.rename(columns={v: k for k, v in INPUT_COLUMNS.items()})
    for n in notes:
        st.caption(n)
else:
    with st.sidebar:
        group = st.selectbox("GICS industry group", group_options, format_func=fmt_group)
    ex = EXAMPLES[group]
    st.subheader("1. Company financials (millions)")
    st.write(f"Enter up to the last 6 fiscal years, oldest first (6 years gives full 5-year averages). "
             f"The table starts with **{ex['ticker']}** as an example.")
    company = st.text_input("Company name or ticker", value=ex["ticker"], key=f"name_{group}")
    default = pd.DataFrame(ex["rows"], columns=list(INPUT_COLUMNS)).rename(columns=INPUT_COLUMNS)
    fin = st.data_editor(default, num_rows="dynamic", width="stretch", hide_index=True, key=f"ed_{group}")
    fin = fin.rename(columns={v: k for k, v in INPUT_COLUMNS.items()}).dropna(subset=["fyear"])

fin = fin.apply(pd.to_numeric, errors="coerce").dropna(subset=["fyear"]).sort_values("fyear")
if len(fin) < 3:
    st.warning("At least 3 fiscal years of statements are needed.")
    st.stop()

model = MODELS[group]
base_year = int(fin["fyear"].iloc[-1])
yr = yearly_ratios(fin)
raw_vals, counts = five_year_averages(yr)
years_used = yr.tail(5)["fyear"]

missing_vars = [VARIABLE_LABELS[v] for v in model["variables"] if pd.isna(raw_vals[v])]
if missing_vars:
    st.error("Can't compute: " + ", ".join(missing_vars) + ". Fill in the missing statement items above.")
    st.stop()
short = [f"{VARIABLE_LABELS[v]} ({counts[v]} yrs)" for v in model["variables"] if counts[v] < 5]
if short:
    st.info(f"Averages use FY{int(years_used.min())}–FY{int(years_used.max())}. Yahoo Finance usually "
            f"provides about 4 years of annual statements, so some averages cover fewer than 5 years: "
            + ", ".join(short) + ".")

# Clip inputs to the range seen in training (same winsorization as the model)
vals, clipped = dict(raw_vals), []
for v in model["variables"]:
    lo, hi = model["bounds"][v]
    if raw_vals[v] < lo or raw_vals[v] > hi:
        clipped.append(VARIABLE_LABELS[v])
    vals[v] = min(max(raw_vals[v], lo), hi)

res = predict(model, vals)
latest_roa = yr["roa"].dropna().iloc[-1] if yr["roa"].notna().any() else np.nan
nx = base_year + 1
name = company or "This company"

# ---------------- Result ----------------
st.subheader(f"2. Forecast ROA for FY{nx}")
st.markdown(
    f"<div style='font-size:1.5rem; line-height:1.5; margin:0.25rem 0 0.75rem'>"
    f"We forecast <b>{name}</b>'s FY{nx} ROA at <b>{res['yhat']:.1%}</b>, and we're 95% confident it will "
    f"land between <b>{res['low']:.1%}</b> and <b>{res['high']:.1%}</b> "
    f"(<b>± {res['margin']:.1%}</b>).</div>", unsafe_allow_html=True)
st.caption(f"95% range = forecast ± 2 × s, where s = {model['s']:.1%} is how far this industry model's "
           f"predictions typically miss actual ROA.")
if clipped:
    st.warning("Outside the training range, so capped at the 1st/99th percentile: " + ", ".join(clipped))

# Range bar: low end to high end, dot at the forecast, marker at last year's actual ROA
fig = go.Figure()
fig.add_trace(go.Scatter(x=[res["low"], res["high"]], y=[0, 0], mode="lines",
                         line=dict(color="rgba(37,125,224,0.35)", width=26), hoverinfo="skip",
                         showlegend=False))
for xv, lab in [(res["low"], "Low"), (res["high"], "High")]:
    fig.add_annotation(x=xv, y=-0.45, text=f"{lab}: <b>{xv:.1%}</b>", showarrow=False, font=dict(size=13))
fig.add_trace(go.Scatter(x=[res["yhat"]], y=[0], mode="markers+text", name=f"FY{nx} forecast",
                         marker=dict(size=18, color=BLUE, line=dict(color="white", width=2)),
                         text=[f"Forecast {res['yhat']:.1%}"], textposition="top center",
                         textfont=dict(size=14), hovertemplate=f"FY{nx} forecast: %{{x:.1%}}<extra></extra>"))
if pd.notna(latest_roa):
    fig.add_trace(go.Scatter(x=[latest_roa], y=[0], mode="markers", name=f"FY{base_year} actual ROA",
                             marker=dict(symbol="diamond", size=14, color="#333333",
                                         line=dict(color="white", width=1.5)),
                             hovertemplate=f"FY{base_year} actual: %{{x:.1%}}<extra></extra>"))
fig.add_vline(x=0, line=dict(color=GRAY, width=1.5, dash="dot"))
span = max(res["high"], latest_roa if pd.notna(latest_roa) else res["high"], 0) - \
       min(res["low"], latest_roa if pd.notna(latest_roa) else res["low"], 0)
pad = 0.10 * span
fig.update_layout(
    title=f"95% range for FY{nx} ROA",
    xaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#eeeeee", zeroline=False,
               range=[min(res["low"], latest_roa if pd.notna(latest_roa) else 0, 0) - pad,
                      max(res["high"], latest_roa if pd.notna(latest_roa) else 0, 0) + pad]),
    yaxis=dict(visible=False, range=[-0.8, 0.8]), height=280, margin=dict(t=50, b=40, l=30, r=30),
    legend=dict(orientation="h", y=-0.25), plot_bgcolor="white")
st.plotly_chart(fig, width="stretch")

contrib = pd.DataFrame({"Variable": [VARIABLE_LABELS[v] for v in res["contrib"]],
                        "Contribution": list(res["contrib"].values())})
fig3 = go.Figure(go.Bar(
    x=contrib["Contribution"], y=contrib["Variable"], orientation="h",
    marker_color=[BLUE if c >= 0 else ORANGE for c in contrib["Contribution"]],
    hovertemplate="%{y}: %{x:+.2%}<extra></extra>"))
fig3.update_layout(title=f"What drives the forecast (intercept {model['coef']['const']:+.2%} not shown)",
                   xaxis_tickformat="+.1%", height=140 + 40 * len(contrib), margin=dict(t=50, b=10),
                   plot_bgcolor="white", bargap=0.45)
fig3.update_xaxes(gridcolor="#eeeeee", zerolinecolor=GRAY)
st.plotly_chart(fig3, width="stretch")

# ---------------- Details ----------------
st.subheader("3. Inputs and model details")
detail = pd.DataFrame({
    "Variable": [VARIABLE_LABELS[v] for v in model["variables"]],
    "Years averaged": [counts[v] for v in model["variables"]],
    "Company value": [raw_vals[v] for v in model["variables"]],
    "Value used": [vals[v] for v in model["variables"]],
    "Coefficient": [model["coef"][v] for v in model["variables"]],
    "Contribution": [res["contrib"][v] for v in model["variables"]],
})
st.dataframe(detail.style.format({"Company value": "{:.4f}", "Value used": "{:.4f}",
                                  "Coefficient": "{:.4f}", "Contribution": "{:+.4f}"}),
             hide_index=True, width="stretch")

with st.expander("Yearly ratios behind the averages"):
    show = yr.tail(5).set_index("fyear")[list(VARIABLE_LABELS)].rename(
        columns={k: v.replace("5-yr avg ", "") for k, v in VARIABLE_LABELS.items()})
    st.dataframe(show.T.style.format("{:.4f}", na_rep="–"), width="stretch")

with st.expander("How this works"):
    eq = f"{model['coef']['const']:.4f} " + " ".join(
        f"{'+' if model['coef'][v] >= 0 else '−'} {abs(model['coef'][v]):.4f} × {VARIABLE_LABELS[v]}"
        for v in model["variables"])
    st.markdown(f"""
**Model for GICS {group} ({model['name']})**

Next-year ROA = {eq}

* Trained on one observation per company: y = FY2025 ROA, x = each ratio averaged over FY2020–2024.
* 5-yr avg ROA is always in the model; the other variables were added by forward selection
  (entry rule p < 0.05) from the remaining 10 candidates.
* R² = {model['r2']:.3f}, adjusted R² = {model['adj_r2']:.3f}, n = {model['n_obs']} companies,
  residual standard error s = {model['s']:.4f}.
* For a forecast, the app averages the same ratios over the company's latest reported years
  (FY{int(years_used.min())}–FY{base_year} here) and predicts FY{nx}.
* **95% range** = forecast ± 2 × s, where s = {model['s']:.4f} is the regression's residual standard
  error (how far its predictions typically miss). About 95% of companies land within 2 standard errors
  of the prediction, assuming the misses are roughly bell-shaped.
""")
