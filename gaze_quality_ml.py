#!/usr/bin/env python3
"""
gaze_quality_ml.py

Compute AOI % fixation features and fit a linear regression model
to predict numeric answer-quality scores.

Usage:
    python gaze_quality_ml.py
"""
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ─── CONFIG ──────────────────────────────────────────────────────────────────
FIX_FILE   = "Eye tracking fixation analysis.xlsx"
QUAL_FILE  = "answer quality participant level analysis.xlsx"
SCORE_COL  = "total"   # column in QUAL_FILE with the numeric score
# ──────────────────────────────────────────────────────────────────────────────

def load_data(fix_fp, qual_fp, score_col):
    # 1) Load spreadsheets
    fix_df  = pd.read_excel(fix_fp)
    qual_df = pd.read_excel(qual_fp)

    # 2) Align the ID column
    if 'PID' in qual_df.columns:
        qual_df.rename(columns={'PID': 'Participant'}, inplace=True)

    # 3) Extract percent-fixation-time columns
    pct_cols = [c for c in fix_df.columns if 'percentage of total fixation time' in c.lower()]
    if not pct_cols:
        raise ValueError("No percentage columns found in fixation file.")
    feat_df = fix_df[['Participant'] + pct_cols].copy()

    # 4) Rename features to Chart1Pct, Chart2Pct, etc.
    rename_map = {}
    for c in pct_cols:
        m = re.search(r'chart\s*(\d+)', c.lower())
        rename_map[c] = f'Chart{m.group(1)}Pct' if m else c
    feat_df.rename(columns=rename_map, inplace=True)

    # 5) Prepare labels
    if score_col not in qual_df.columns:
        raise ValueError(f"Score column '{score_col}' not found. Available: {qual_df.columns.tolist()}")
    labels = qual_df[['Participant', score_col]].rename(columns={score_col: 'QualityScore'})

    # 6) Merge and return
    return pd.merge(feat_df, labels, on='Participant')

def main():
    # Load and merge
    df = load_data(FIX_FILE, QUAL_FILE, SCORE_COL)

    # Split out X and y
    X = df.drop(columns=['Participant', 'QualityScore'])
    y = df['QualityScore']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}")
    print(f"Test R^2: {r2_score(y_test, y_pred):.3f}\n")

    # Show coefficients
    coef_df = (
        pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
          .assign(AbsCoeff=lambda d: d['Coefficient'].abs())
          .sort_values('AbsCoeff', ascending=False)
          .drop(columns='AbsCoeff')
    )
    print("Feature coefficients:")
    print(coef_df.to_string(index=False))

if __name__ == '__main__':
    main()
