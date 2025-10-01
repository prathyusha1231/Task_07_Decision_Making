# Research Task 7: NCAA Women’s Lacrosse Analysis (2022–2023)

## Overview
This repository contains the full workflow and report for **Research Task 7**, which extends the descriptive analysis from Task 5 into a reproducible, ethical, decision-support framework.  
The project uses **NCAA Division I Women’s Lacrosse 2022–2023** team-level statistics to identify key drivers of win percentage, assess fairness across conferences, and generate risk-tiered coaching recommendations.


## Requirements

- python 3.11+
- Install packages:
pip install -r requirements.txt

## How to Reporoduce

1. Run descriptive stats, correlations, and figures → outputs/

python scripts/analyze_scripts.py

2. Run leakage/proxy checks → outputs/leakage_report.txt + suspicious_corrs.csv

python scripts/leakage_review.py

3. Run LLM evaluation (GPT-4 Turbo) → outputs/llm_evaluation_*.{csv,txt}

python scripts/openai_script.py

## Repository Structure
```text
.
├── scripts/
│   ├── analyze_scripts.py        # descriptive stats, correlations, plots
│   ├── leakage_review.py         # leakage/proxy feature checks
│   └── openai_script.py          # GPT-4 Turbo evaluation
├── outputs/
│   ├── lacrosse_statistics.txt   # descriptives + correlations
│   ├── leakage_report.txt        # flagged leakage results
│   ├── suspicious_corrs.csv      # suspiciously high correlations
│   ├── llm_evaluation_results_gpt-4-turbo_20250730_221250.csv
│   ├── llm_evaluation_summary_gpt-4-turbo_20250730_221250.txt
│   └── images/
│       ├── correlation_heatmap.png
│       ├── defensive_analysis.png
│       ├── goals_comparison.png
│       ├── shot_efficiency.png
│       ├── top_teams_win_percentage.png
│       └── win_percentage_distribution.png
├── report/
│   └── Research_Task_07.pdf      # final written report
├── .gitignore
└── README.md

