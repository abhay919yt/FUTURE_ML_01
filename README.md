Sales & Demand Forecasting Project
A monthly sales forecasting pipeline built for internship deliverable requirements: data cleaning, time-based feature engineering, regression/time-series forecasting, error analysis, and business-friendly visual output.

1. Get the dataset
This project is built around the Superstore Sales Dataset (Kaggle), but any transactional sales dataset with a date column and a sales/quantity column will work with minor tweaks to column names in src/data_cleaning.py.

Download "Sample Superstore" from Kaggle: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final (search "Superstore Sales Dataset" if that link changes)
Save the CSV as: data/superstore.csv
Expected key columns: Order Date, Sales, Category, Sub-Category, Region, Segment. If your file uses different column names, update the COLUMN_MAP at the top of src/data_cleaning.py.

2. Install dependencies
pip install -r requirements.txt
3. Run the pipeline
Open notebooks/main.ipynb and run cells top to bottom, OR run the scripts directly:

python src/data_cleaning.py
python src/feature_engineering.py
python src/modeling.py
python src/evaluation.py
python src/visualization.py
Each stage saves its output into outputs/ so you can inspect intermediate results, and the final forecast chart + summary table land in outputs/ for pasting into Excel/Power BI or your report.

Project structure
sales_forecasting/
├── data/
│   └── superstore.csv          <- you add this
├── src/
│   ├── data_cleaning.py        <- load, clean, aggregate to monthly
│   ├── feature_engineering.py  <- lag/rolling/calendar features
│   ├── modeling.py             <- baseline + SARIMA + XGBoost/RF models
│   ├── evaluation.py           <- MAE/RMSE/MAPE + residual analysis
│   └── visualization.py        <- actual vs forecast chart, summary table
├── notebooks/
│   └── main.ipynb              <- end-to-end walkthrough
├── outputs/                    <- all generated CSVs and charts land here
└── requirements.txt
Notes on approach
Granularity: monthly (aggregated from transaction-level data)
Baseline: naive forecast (same month last year) — always compare fancier models against this to prove they add value
Models compared: SARIMA (classical, good for single series with seasonality) vs XGBoost/RandomForest regression (uses lag + calendar features, generalizes better across categories/regions)
Validation: time-ordered train/test split (last 6 months held out) — never random shuffle for time series
Metrics: MAPE (most business-interpretable), plus MAE and RMSE
