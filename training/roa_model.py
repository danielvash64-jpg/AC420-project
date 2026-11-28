"""
roa_model.py
============
Shared code used by the five industry training scripts (train_2010.py, ...).

It does four things:
  1. Builds the 11 candidate predictor variables and the ROA target from the
     raw Compustat columns in complete_companies_2014-2025.xlsx.
  2. Winsorizes every variable at the 1st / 99th percentile.
  3. Runs p-value-based forward selection with OLS, always keeping the
     5-year average ROA in the model.
  4. Writes a Markdown report plus a JSON file with the final coefficients
     that the Streamlit app uses.

Design: one observation per company.
  y  = ROA in FY2025
  x  = each variable averaged over the five fiscal years FY2020-FY2024

Yearly definitions (year t), each then averaged over FY2020-2024:
  ROA                   = NI_t / average(AT_t, AT_t-1)
  Revenue growth        = REVT_t / REVT_t-1 - 1        (same for GP and SG&A)
  GM                    = GP / REVT
  OM                    = OIADP / REVT
  SG&A / revenue        = XSGA / REVT
  Asset turnover        = REVT / average(AT_t, AT_t-1)
  GM change             = GM_t - GM_t-1
  OM change             = OM_t - OM_t-1
  OCF / average assets  = OANCF / average(AT_t, AT_t-1)
The FY2020 values use FY2019 data for prior-year assets, growth and changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "complete_companies_2014-2025.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

INDUSTRY_NAMES = {
    2010: "Capital Goods",
    2550: "Consumer Discretionary Distribution & Retail",
    3520: "Pharmaceuticals, Biotechnology & Life Sciences",
    4510: "Software & Services",
    4530: "Semiconductors & Semiconductor Equipment",
}

# Candidate pool (all are 5-year averages), in the order given in the assignment
VARIABLES = {
    "roa": "5-yr avg ROA",
    "rev_growth": "5-yr avg revenue growth",
    "gm": "5-yr avg gross margin (GM)",
    "om": "5-yr avg operating margin (OM)",
    "sga_rev": "5-yr avg SG&A / revenue",
    "gp_growth": "5-yr avg gross profit growth",
    "asset_turn": "5-yr avg asset turnover",
    "sga_growth": "5-yr avg SG&A growth",
    "gm_change": "5-yr avg GM change",
    "om_change": "5-yr avg OM change",
    "ocf_assets": "5-yr avg OCF / avg assets",
}
TARGET = "roa_target"

TARGET_YEAR = 2025        # y = ROA in this fiscal year
WINDOW = 5                # x = averages over the 5 fiscal years before it (FY2020-2024)
MIN_YEARS = 3             # a company needs at least 3 valid years of a variable
ENTRY_P_VALUE = 0.05      # a variable enters only if its p-value is below this
WINSOR_PCT = (0.01, 0.99)  # winsorize each variable at these percentiles
FORCED = ["roa"]          # always kept in the model; forward selection picks from the rest
CONF_LEVELS = (0.80, 0.90, 0.95, 0.99)


# --------------------------------------------------------------------------
# 1. Variable construction
# --------------------------------------------------------------------------
def _safe_div(a, b):
    b = b.where(b != 0)
    return a / b


def _growth(s: pd.Series) -> pd.Series:
    """Year-over-year growth; undefined when last year's value is <= 0."""
    prev = s.shift(1)
    return (s - prev) / prev.where(prev > 0)


def yearly_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """One company's statements (one row per fiscal year, oldest first) -> yearly ratios."""
    df = df.sort_values("fyear").reset_index(drop=True)
    out = pd.DataFrame({"fyear": df["fyear"]})
    avg_at = (df["at"] + df["at"].shift(1)) / 2
    out["roa"] = _safe_div(df["ni"], avg_at)
    out["gm"] = _safe_div(df["gp"], df["revt"])
    out["om"] = _safe_div(df["oiadp"], df["revt"])
    out["sga_rev"] = _safe_div(df["xsga"], df["revt"])
    out["asset_turn"] = _safe_div(df["revt"], avg_at)
    out["ocf_assets"] = _safe_div(df["oancf"], avg_at)
    out["gm_change"] = out["gm"] - out["gm"].shift(1)
    out["om_change"] = out["om"] - out["om"].shift(1)
    out["rev_growth"] = _growth(df["revt"])
    out["gp_growth"] = _growth(df["gp"])
    out["sga_growth"] = _growth(df["xsga"])
    return out.replace([np.inf, -np.inf], np.nan)


def build_variables(raw: pd.DataFrame, target_year: int = TARGET_YEAR) -> pd.DataFrame:
    """One row per company: 5-year averages of each ratio (FY t-5 .. t-1) and ROA in t."""
    rows = []
    first = target_year - WINDOW
    for gvkey, firm in raw.groupby("gvkey"):
        yr = yearly_ratios(firm)
        window = yr[(yr["fyear"] >= first) & (yr["fyear"] <= target_year - 1)]
        avg = window[list(VARIABLES)].apply(
            lambda s: s.mean() if s.notna().sum() >= MIN_YEARS else np.nan)
        target = yr.loc[yr["fyear"] == target_year, "roa"]
        rows.append({"gvkey": gvkey, "tic": firm["tic"].iloc[0], **avg.to_dict(),
                     TARGET: target.iloc[0] if len(target) else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Winsorization
# --------------------------------------------------------------------------
def winsor_bounds(df: pd.DataFrame, cols) -> dict:
    return {c: (float(df[c].quantile(WINSOR_PCT[0])), float(df[c].quantile(WINSOR_PCT[1]))) for c in cols}


def apply_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        out[c] = out[c].clip(lo, hi)
    return out


# --------------------------------------------------------------------------
# 3. Forward selection
# --------------------------------------------------------------------------
def forward_selection(df: pd.DataFrame, candidates, forced=(), target=TARGET, threshold=ENTRY_P_VALUE):
    """Start from the forced variables. At each step try adding every remaining
    variable; add the one with the lowest p-value if that p-value is below the
    threshold. Stop otherwise."""
    selected, steps = list(forced), []
    remaining = [c for c in candidates if c not in forced]
    while remaining:
        rows = []
        for var in remaining:
            X = sm.add_constant(df[selected + [var]])
            fit = sm.OLS(df[target], X).fit()
            rows.append({
                "variable": var,
                "coef": fit.params[var],
                "t_stat": fit.tvalues[var],
                "p_value": fit.pvalues[var],
                "adj_r2": fit.rsquared_adj,
                "aic": fit.aic,
            })
        table = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
        best = table.iloc[0]
        added = best["p_value"] < threshold
        steps.append({"step": len(steps) + 1, "already_in": list(selected), "table": table,
                      "added": best["variable"] if added else None})
        if not added:
            break
        selected.append(best["variable"])
        remaining.remove(best["variable"])
    return selected, steps


# --------------------------------------------------------------------------
# 4. Testing, export and reporting
# --------------------------------------------------------------------------
def export_model(fit, selected, bounds, n_obs, industry) -> dict:
    cols = ["const"] + selected
    t_crit = {f"{lvl:.2f}": float(stats.t.ppf(0.5 + lvl / 2, fit.df_resid)) for lvl in CONF_LEVELS}
    return {
        "ggroup": industry,
        "name": INDUSTRY_NAMES[industry],
        "variables": selected,
        "coef": {c: float(fit.params[c]) for c in cols},
        "cov": [[float(fit.cov_params().loc[a, b]) for b in cols] for a in cols],
        "sigma2": float(fit.scale),
        "df_resid": float(fit.df_resid),
        "t_crit": t_crit,
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "n_obs": int(n_obs),
        "bounds": {c: list(bounds[c]) for c in selected},
    }


def _md_table(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def run_industry(industry: int, data_file: Path = DATA_FILE, out_dir: Path = OUTPUT_DIR) -> dict:
    raw = pd.read_excel(data_file)
    raw = raw[raw["ggroup"] == industry]
    cols = list(VARIABLES) + [TARGET]
    allvars = build_variables(raw)
    train = allvars.dropna(subset=cols)

    # Winsorize at the 1st / 99th percentile
    bounds = winsor_bounds(train, cols)
    train_w = apply_bounds(train, bounds)

    # Forward selection, with 5-year average ROA always in the model
    selected, steps = forward_selection(train_w, list(VARIABLES), forced=FORCED)
    fit = sm.OLS(train_w[TARGET], sm.add_constant(train_w[selected])).fit()
    model = export_model(fit, selected, bounds, len(train_w), industry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"model_{industry}.json").write_text(json.dumps(model, indent=2))

    # ---- Markdown report
    first, last = TARGET_YEAR - WINDOW, TARGET_YEAR - 1
    name = INDUSTRY_NAMES[industry]
    L = []
    L.append(f"# Forward selection results: GICS {industry} ({name})\n")
    L.append(f"Generated by `training/train_{industry}.py`. Goal: predict a company's ROA "
             f"(NI / average total assets) from the 5-year averages of its financial ratios.\n")

    L.append("## 1. Sample and design\n")
    L.append(_md_table(pd.DataFrame({
        "": ["Companies in industry group", "Companies with all variables (used in regression)"],
        "Count": [raw["gvkey"].nunique(), len(train)],
    })))
    L.append(f"\n* **y:** ROA in FY{TARGET_YEAR}\n"
             f"* **x:** each ratio averaged over FY{first}–FY{last} (5 years; growth rates and changes for "
             f"FY{first} use FY{first - 1} data). A company needs at least {MIN_YEARS} valid years of a ratio.\n"
             f"* One observation per company. All variables are winsorized at the 1st and 99th percentiles.\n"
             f"* In the app, the same averages are computed from a company's most recent reported years "
             f"to forecast the next fiscal year's ROA.\n")

    L.append("## 2. Candidate variables (after winsorizing)\n")
    desc = train_w[cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.insert(0, f"Correlation with FY{TARGET_YEAR} ROA", train_w[cols].corr()[TARGET])
    desc.index = [VARIABLES.get(c, f"**FY{TARGET_YEAR} ROA (target)**") for c in desc.index]
    L.append(desc.reset_index().rename(columns={"index": "Variable", "50%": "median"}).to_markdown(index=False, floatfmt=".4f"))

    L.append(f"\n## 3. Forward selection steps (entry rule: p-value < {ENTRY_P_VALUE})\n")
    L.append("**5-yr avg ROA is always included.** Starting from that model, every remaining variable is "
             "added one at a time. The variable with the lowest p-value enters if p < 0.05; otherwise "
             "selection stops.\n")
    for s in steps:
        base = ", ".join(VARIABLES[v] for v in s["already_in"]) or "intercept only"
        L.append(f"### Step {s['step']} (current model: {base})\n")
        t = s["table"].copy()
        t["variable"] = t["variable"].map(VARIABLES)
        t.columns = ["Variable tried", "Coefficient", "t-stat", "p-value", "Adj. R²", "AIC"]
        L.append(_md_table(t))
        if s["added"]:
            L.append(f"\n**Added: {VARIABLES[s['added']]}**\n")
        else:
            L.append(f"\n**Stop: no remaining variable has p < {ENTRY_P_VALUE}.**\n")

    L.append("## 4. Final model (coefficients used in the app)\n")
    coef = pd.DataFrame({
        "Variable": ["Intercept"] + [VARIABLES[v] for v in selected],
        "Coefficient": fit.params.values,
        "Std. error": fit.bse.values,
        "t-stat": fit.tvalues.values,
        "p-value": fit.pvalues.values,
    })
    L.append(_md_table(coef))
    eq = " ".join([f"{fit.params['const']:.4f}"] +
                  [f"{'+' if fit.params[v] >= 0 else '−'} {abs(fit.params[v]):.4f} × {VARIABLES[v]}"
                   for v in selected])
    L.append(f"\n**Next-year ROA = {eq}**\n")
    L.append(f"R² = {fit.rsquared:.4f}, adjusted R² = {fit.rsquared_adj:.4f}, n = {int(fit.nobs)}, "
             f"residual std. error = {np.sqrt(fit.scale):.4f}, F-test p-value = {fit.f_pvalue:.2e}\n")

    L.append("## 5. Prediction and confidence intervals\n")
    L.append("For a new company with inputs x, the app reports:\n")
    L.append("* **Prediction interval** (where that company's ROA should land): ŷ ± t × √(s² + x′Vx)")
    L.append("* **Confidence interval** (average ROA of companies with those ratios): ŷ ± t × √(x′Vx)\n")
    L.append("where s² is the residual variance and V is the coefficient covariance matrix from this regression. "
             "t critical values (df = %d):\n" % int(fit.df_resid))
    L.append(_md_table(pd.DataFrame({"Confidence level": [f"{float(k):.0%}" for k in model["t_crit"]],
                                     "t": list(model["t_crit"].values())})))

    (out_dir / f"results_{industry}.md").write_text("\n".join(L))
    print("\n".join(L))
    return model
