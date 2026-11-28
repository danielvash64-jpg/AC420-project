"""
Training script for GICS industry group 4510.

Run from the repository root:
    python training/train_4510.py

It builds the 11 candidate variables, runs forward selection on FY2024
ratios (predicting ROA 2025) with previous year ROA always included, and writes:
    training/results/results_4510.md   (step-by-step report, renders on GitHub)
    training/results/model_4510.json   (final coefficients used by app.py)
"""
from roa_model import run_industry

INDUSTRY = 4510

if __name__ == "__main__":
    run_industry(INDUSTRY)
