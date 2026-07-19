"""
run_all.py
----------
Runs the full pipeline end to end:
  data cleaning -> feature engineering -> modeling -> evaluation -> visualization

Usage:
    python run_all.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data_cleaning, feature_engineering, modeling, evaluation, visualization


def main():
    print("=" * 60)
    print("STEP 1/5: Data cleaning")
    print("=" * 60)
    data_cleaning.run()

    print("\n" + "=" * 60)
    print("STEP 2/5: Feature engineering")
    print("=" * 60)
    feature_engineering.run()

    print("\n" + "=" * 60)
    print("STEP 3/5: Modeling")
    print("=" * 60)
    modeling.run()

    print("\n" + "=" * 60)
    print("STEP 4/5: Evaluation")
    print("=" * 60)
    evaluation.run()

    print("\n" + "=" * 60)
    print("STEP 5/5: Visualization")
    print("=" * 60)
    visualization.run()

    print("\nDone. Check the outputs/ folder for charts, tables, and metrics.")


if __name__ == "__main__":
    main()
