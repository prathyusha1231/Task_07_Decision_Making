import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
DATA_PATH = "lacrosse_women_ncaa_div1_2022_2023.csv"  
TARGET = "win_pctg"

# ------------- Helper: identify suspicious columns by name -------------
LEAKY_NAME_PATTERNS = [
    r"^win", r"_win", r"wins?", r"loss", r"record", r"ranking", r"seed",
    r"playoff", r"standing", r"pctg$"  # catches pctg columns; we'll allow some explicitly later
]

IDENTIFIER_PATTERNS = [
    r"^team$", r"^team_id$", r"^conference$", r"id$", r"_id$"
]

# ------------- Helper: pick actionable features (coach levers) -------------
ACTIONABLE_WHITELIST = [
    "draw_pctg",
    "assists_per_game",
    "shot_pctg",
    "sog_per_game",
    "shots_per_game",
    "turnovers_per_game",
    "save_pctg",
    "caused_turnovers_per_game",
    "clearing_pctg",
    "ground_balls_per_game",
    "ride_success_pctg"
]

def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # strip team/conference strings just in case
    if "Team" in df.columns:
        df["Team"] = df["Team"].astype(str).str.strip()
    return df

def cols_matching(patterns, columns):
    out = set()
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        out |= {c for c in columns if rx.search(c)}
    return list(sorted(out))

def near_duplicate_pairs(df, threshold=0.98):
    """Return pairs of columns with |corr| >= threshold."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr().abs()
    pairs = []
    for c1, c2 in combinations(corr.columns, 2):
        if corr.loc[c1, c2] >= threshold:
            pairs.append((c1, c2, corr.loc[c1, c2]))
    return sorted(pairs, key=lambda x: -x[2])

def possible_linear_combo(df, a, b, tol=1e-6):
    """Check if column 'a' ~ sum of some other columns (simple heuristic)."""
    # Example heuristic: points_per_game ≈ goals_per_game + assists_per_game
    # If 'a' and 'b' exist, test a - b ~ 0
    if a in df.columns and b in df.columns:
        diff = (df[a] - df[b]).abs().fillna(np.inf)
        return (diff.max() < tol, float(diff.max()))
    return (False, float("inf"))

def pick_numeric(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def baseline_model_cv(X, y, cv_splits=5):
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("lin", LinearRegression())
    ])
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(pipe, X, y, scoring="r2", cv=cv)
    return scores

def main():
    out_dir = Path("./")
    report_path = out_dir / "leakage_report.txt"
    corr_csv_path = out_dir / "suspicious_corrs.csv"

    df = load_data(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in CSV.")

    # Identify identifiers & blatantly leaky columns by name
    all_cols = df.columns.tolist()
    id_like = cols_matching(IDENTIFIER_PATTERNS, all_cols)
    leaky_name_hits = cols_matching(LEAKY_NAME_PATTERNS, all_cols)

    #*allow* some pctg columns that are legitimate levers 
    allow_if_in_actionable = set(ACTIONABLE_WHITELIST)
    blatantly_leaky = [c for c in leaky_name_hits if c != TARGET and c not in allow_if_in_actionable]

    # Check near-duplicate columns (proxy leakage risk)
    dup_pairs = near_duplicate_pairs(df)

    # Check obvious linear-combo proxy: points_per_game ~ goals_per_game + assists_per_game

    linear_proxy_findings = []
    if "points_per_game" in df.columns:
        if "goals_per_game" in df.columns and "assists_per_game" in df.columns:
            df["_goals_plus_assists"] = df["goals_per_game"] + df["assists_per_game"]
            is_combo, max_abs_diff = possible_linear_combo(df, "points_per_game", "_goals_plus_assists")
            linear_proxy_findings.append({
                "lhs": "points_per_game",
                "rhs": "goals_per_game + assists_per_game",
                "is_near_equal": bool(is_combo),
                "max_abs_diff": float(max_abs_diff)
            })
            df.drop(columns=["_goals_plus_assists"], errors="ignore", inplace=True)

    # Build feature sets
    numeric_cols = pick_numeric(df)
    y = df[TARGET].values

    # Drop target and obvious IDs from numeric feature set
    drop_set = set([TARGET]) | set(id_like)
    X_all_cols = [c for c in numeric_cols if c not in drop_set]

    # Actionable subset (intersection of whitelist and available numeric columns)
    X_actionable_cols = [c for c in ACTIONABLE_WHITELIST if c in numeric_cols and c != TARGET]

    # Train baseline A (All numeric minus target/IDs) and B (Actionable)
    X_all = df[X_all_cols].dropna()
    y_all = df.loc[X_all.index, TARGET].values

    X_act = df[X_actionable_cols].dropna()
    y_act = df.loc[X_act.index, TARGET].values

    scores_all = baseline_model_cv(X_all, y_all, cv_splits=5)
    scores_act = baseline_model_cv(X_act, y_act, cv_splits=5)

    # Correlations between suspicious columns and target
    suspicious_cols = sorted(set([c for pair in dup_pairs for c in pair[:2]]) | set(blatantly_leaky))
    suspicious_cols = [c for c in suspicious_cols if c in numeric_cols and c != TARGET]
    corr_rows = []
    for c in suspicious_cols:
        corr = df[[c, TARGET]].corr().iloc[0, 1]
        corr_rows.append({"column": c, "corr_with_win_pctg": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_win_pctg", ascending=False)
    corr_df.to_csv(corr_csv_path, index=False)

    # Write report
    with report_path.open("w", encoding="utf-8") as f:
        f.write("LEAKAGE REVIEW REPORT\n")
        f.write("=====================================\n\n")
        f.write(f"Dataset: {DATA_PATH}\n")
        f.write(f"Target: {TARGET}\n\n")

        f.write("1) Identifier-like columns (dropped from modeling):\n")
        f.write(f"   {id_like if id_like else 'None found'}\n\n")

        f.write("2) Name-pattern leakage candidates (excluding actionable pctg columns):\n")
        f.write(f"   {blatantly_leaky if blatantly_leaky else 'None found'}\n\n")

        f.write("3) Near-duplicate numeric pairs (|corr| >= 0.98):\n")
        if dup_pairs:
            for a, b, c in dup_pairs[:20]:
                f.write(f"   {a} ~ {b} (|r|={c:.3f})\n")
        else:
            f.write("   None\n")
        f.write("\n")

        if linear_proxy_findings:
            f.write("4) Linear-combo proxy checks:\n")
            for rec in linear_proxy_findings:
                f.write(f"   {rec['lhs']} ≈ {rec['rhs']}? "
                        f"is_near_equal={rec['is_near_equal']} max_abs_diff={rec['max_abs_diff']:.6f}\n")
            f.write("\n")

        f.write("5) Cross-validated R^2 (mean ± std):\n")
        f.write(f"   Model A (All numeric minus target/IDs): {scores_all.mean():.3f} ± {scores_all.std():.3f}\n")
        f.write(f"   Model B (Actionable-only levers):       {scores_act.mean():.3f} ± {scores_act.std():.3f}\n")
        f.write("   Interpretation: If Model A >> Model B, the gap likely reflects proxy/outcome-like features\n")
        f.write("   inflating performance. Use Model B for coaching recommendations to avoid leakage.\n\n")

        f.write("6) Suspicious columns vs target correlation saved to: suspicious_corrs.csv\n")
        f.write("   Review high |corr| columns—especially those that summarize outcomes rather than inputs.\n\n")

        summary = {
            "id_like": id_like,
            "blatantly_leaky": blatantly_leaky,
            "near_duplicate_pairs_count": len(dup_pairs),
            "model_a_r2_mean": float(scores_all.mean()),
            "model_a_r2_std": float(scores_all.std()),
            "model_b_r2_mean": float(scores_act.mean()),
            "model_b_r2_std": float(scores_act.std()),
            "linear_proxy_checks": linear_proxy_findings
        }
        f.write("JSON_SUMMARY:\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n")

    print(f"[OK] Leakage review complete.\n- Report: {report_path}\n- Suspicious correlations: {corr_csv_path}")
    

if __name__ == "__main__":
    main()
