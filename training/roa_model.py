"""
roa_model.py
============
Shared code used by the five industry training scripts (train_2010.py, ...).

It does four things:
  1. Builds the 11 candidate predictor variables and the future-ROA target
     from the raw Compustat columns in complete_companies_2014-2025.xlsx.
  2. Winsorizes every variable at the 1st / 99th percentile.
  3. Runs p-value-based forward selection with OLS, always keeping
     previous year ROA in the model.
  4. Writes a Markdown report plus a JSON file with the final coefficients
     that the Streamlit app uses.

Training sample: FY2024 ratios -> ROA 2025 (the most recent year with a
known outcome; the 5-year averages reach back to FY2019). The app then uses
FY2025 ratios to predict ROA for 2026.

Definitions (year t = the most recent fiscal year we have data for):
  ROA_t                 = NI_t / average(AT_t, AT_t-1)
  Target (future ROA)   = ROA_t+1
  Previous year ROA     = ROA_t   (the year before the year being predicted)
  5-yr avg growth of X  = mean of the five yearly growth rates X_t-4/X_t-5-1 ... X_t/X_t-1-1
  GM                    = GP / REVT
  OM                    = OIADP / REVT
  SG&A / revenue        = XSGA / REVT
  Asset turnover        = REVT / average(AT_t, AT_t-1)
  GM change             = GM_t - GM_t-1
  OM change             = OM_t - OM_t-1
  OCF / average assets  = OANCF / average(AT_t, AT_t-1)
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

# Candidate pool, in the order given in the assignment
VARIABLES = {
    "roa_prev": "Previous year ROA",
    "rev_growth_5y": "5-yr avg revenue growth",
    "gm": "Gross margin (GM)",
    "om": "Operating margin (OM)",
    "sga_rev": "SG&A / revenue",
    "gp_growth_5y": "5-yr avg gross profit growth",
    "asset_turn": "Asset turnover (REVT / avg AT)",
    "sga_growth_5y": "5-yr avg SG&A growth",
    "gm_change": "GM change",
    "om_change": "OM change",
    "ocf_assets": "OCF / avg assets",
}
TARGET = "roa_next"

ENTRY_P_VALUE = 0.05      # a variable enters only if its p-value is below this
WINSOR_PCT = (0.01, 0.99)  # winsorize each variable at these percentiles
# Train on the most recent year only: FY2024 ratios -> ROA 2025.
# (The 5-year averages use data back to FY2019.)
TRAIN_YEARS = (2024,)
FORCED = ["roa_prev"]     # always kept in the model; forward selection picks from the rest
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


def build_variables(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["gvkey", "fyear"]).copy()
    g = df.groupby("gvkey", group_keys=False)

    avg_at = (df["at"] + g["at"].shift(1)) / 2
    df["roa"] = _safe_div(df["ni"], avg_at)
    df["roa_prev"] = df["roa"]
    df["gm"] = _safe_div(df["gp"], df["revt"])
    df["om"] = _safe_div(df["oiadp"], df["revt"])
    df["sga_rev"] = _safe_div(df["xsga"], df["revt"])
    df["asset_turn"] = _safe_div(df["revt"], avg_at)
    df["ocf_assets"] = _safe_div(df["oancf"], avg_at)

    g = df.groupby("gvkey", group_keys=False)
    df["gm_change"] = df["gm"] - g["gm"].shift(1)
    df["om_change"] = df["om"] - g["om"].shift(1)

    for col, name in [("revt", "rev_growth_5y"), ("gp", "gp_growth_5y"), ("xsga", "sga_growth_5y")]:
        yoy = g[col].apply(_growth)
        # average of the last five yearly growth rates (needs at least 3 valid ones)
        df[name] = yoy.groupby(df["gvkey"]).transform(lambda s: s.rolling(5, min_periods=3).mean())
        # require the full 5-year window to exist (6 years of data)
        df.loc[g.cumcount() < 5, name] = np.nan

    df[TARGET] = df.groupby("gvkey")["roa"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


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
        "bounds": {c: list(bounds[c]) for c in selected + ["roa_prev"]},
    }


def _md_table(df: pd.DataFrame, floatfmt=".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def run_industry(industry: int, data_file: Path = DATA_FILE, out_dir: Path = OUTPUT_DIR) -> dict:
    raw = pd.read_excel(data_file)
    raw = raw[raw["ggroup"] == industry]
    allvars = build_variables(raw)
    cols = list(VARIABLES) + [TARGET]
    train = allvars[allvars["fyear"].isin(TRAIN_YEARS)].dropna(subset=cols)

    # Winsorize at the 1st / 99th percentile
    bounds = winsor_bounds(train, cols)
    train_w = apply_bounds(train, bounds)

    # Forward selection, with previous year ROA always in the model
    selected, steps = forward_selection(train_w, list(VARIABLES), forced=FORCED)
    fit = sm.OLS(train_w[TARGET], sm.add_constant(train_w[selected])).fit()
    model = export_model(fit, selected, bounds, len(train_w), industry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"model_{industry}.json").write_text(json.dumps(model, indent=2))

    # ---- Markdown report
    name = INDUSTRY_NAMES[industry]
    L = []
    L.append(f"# Forward selection results: GICS {industry} ({name})\n")
    L.append("Generated by `training/train_%d.py`. Goal: predict next year's ROA "
             "(NI / average total assets) from this year's financial ratios.\n" % industry)

    L.append("## 1. Sample\n")
    L.append(_md_table(pd.DataFrame({
        "": ["Companies in industry group", "Companies used (FY2024 ratios predicting ROA 2025)"],
        "Count": [raw["gvkey"].nunique(), len(train)],
    })))
    L.append("\nThe model is trained on the most recent year with a known outcome: FY2024 ratios predict "
             "2025 ROA. The 5-year average growth variables use data back to FY2019. All variables are winsorized at the 1st and 99th percentiles. "
             "The finished model takes FY2025 ratios to predict 2026 ROA.\n")

    L.append("## 2. Candidate variables (after winsorizing)\n")
    desc = train_w[cols].describe().T[["mean", "std", "min", "50%", "max"]]
    desc.insert(0, "Correlation with future ROA", train_w[cols].corr()[TARGET])
    desc.index = [VARIABLES.get(c, "**Future ROA (target)**") for c in desc.index]
    L.append(desc.reset_index().rename(columns={"index": "Variable", "50%": "median"}).to_markdown(index=False, floatfmt=".4f"))

    L.append(f"\n## 3. Forward selection steps (entry rule: p-value < {ENTRY_P_VALUE})\n")
    L.append("**Previous year ROA is always included.** Starting from that model, every remaining variable is "
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
    L.append(f"\n**Future ROA = {eq}**\n")
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
