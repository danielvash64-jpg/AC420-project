"""
Training script for GICS industry group 3520.

Run from the repository root:
    python training/train_3520.py

It averages the 11 candidate variables over FY2020-2024, runs forward selection
to predict FY2025 ROA (5-yr avg ROA always included), and writes:
    training/results/results_3520.md   (step-by-step report, renders on GitHub)
    training/results/model_3520.json   (final coefficients used by app.py)
"""
from roa_model import run_industry

INDUSTRY = 3520

if __name__ == "__main__":
    run_industry(INDUSTRY)
