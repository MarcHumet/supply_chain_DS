
"""
h2o_forecast_demand.py
----------------------
Monthly product demand forecasting using H2O AutoML.

This script:
 - Loads processed supply chain data
 - Selects and aggregates relevant features for monthly product-level demand
 - Trains H2O AutoML regression models
 - Evaluates and prints leaderboard and predictions

Models considered by H2O AutoML include:
 - Gradient Boosting Machines (GBM)
 - Distributed Random Forest (DRF)
 - Deep Learning (Neural Networks)
 - Generalized Linear Models (GLM)
 - XGBoost (if enabled)
 - Stacked Ensembles
 - Extremely Randomized Trees (XRT)
 - RuleFit

To run this script, ensure you have H2O installed and properly configured in your Python environment. 
Adjust the DATA_PATH variable to point to your processed dataset. 
If in proper environment, run python h2o_forecast_demand.py

python run h2o will automatically utilize available CPU cores for training.
Author: Marc Humet
Date: 2026-04-16
"""

import pandas as pd
import h2o
from h2o.automl import H2OAutoML


# --- CONFIGURATION ---
DATA_PATH = "/home/marc/project/supply_chain_DS/dataco_supply_chain/data/processed/demand_forecast_dataset.csv"
TARGET = "Order Item Quantity" # Change to 'Order Item Quantity' to predict units
GROUP_COLS = [
	'month',
	'year',
	'Product Card Id',
	'Product Price',
	'Category Name',
	'Department Name',
	'Market',
]
REQUIRED_FEATURES = GROUP_COLS + [TARGET]


# --- 1. LOAD DATA ---
df = pd.read_csv(DATA_PATH)
df = df[REQUIRED_FEATURES].copy()

# --- 2. AGGREGATE MONTHLY DEMAND PER PRODUCT ---
df_agg = df.groupby(GROUP_COLS, dropna=False)[TARGET].sum().reset_index()

# --- 3. INITIALIZE H2O ---
h2o.init()

# --- 4. LOAD DATA INTO H2O ---
hf = h2o.H2OFrame(df_agg)

# --- 5. FEATURE ENGINEERING ---
# Convert categorical columns to factors
for col in ['Product Card Id', 'Category Name', 'Department Name', 'Market']:
	hf[col] = hf[col].asfactor()

# --- 6. TRAIN/TEST SPLIT ---
train, test = hf.split_frame(ratios=[0.8], seed=42)

# --- 7. DEFINE FEATURES ---
features = [c for c in hf.columns if c != TARGET]

# --- 8. RUN H2O AUTOML ---
aml = H2OAutoML(max_runtime_secs=600, seed=1)
aml.train(x=features, y=TARGET, training_frame=train)

# --- 9. EVALUATE ---
lb = aml.leaderboard
print("\nH2O AutoML Leaderboard:\n", lb)

perf = aml.leader.model_performance(test)
print("\nTest set performance:")
print(perf)

# --- 10. PREDICT ---
preds = aml.leader.predict(test)
print("\nSample predictions:")
print(preds.head())

# --- 11. SHUTDOWN H2O (optional) ---
# h2o.shutdown(prompt=False)
h2o.init()

data_path = "/home/marc/project/supply_chain_DS/dataco_supply_chain/data/processed/demand_forecast_dataset.csv"
df_h2o = h2o.import_file(data_path)
train, test = df_h2o.split_frame(ratios=[0.8], seed=1234)

target = 'Order Item Quantity'  # or your demand column
features = [col for col in df_h2o.columns if col not in [target, 'shipping date (DateOrders)']]

aml = H2OAutoML(max_runtime_secs=600, seed=1)
aml.train(x=features, y=target, training_frame=train)
model_path = h2o.save_model(model=aml.leader, path=".", force=True)
print(f"Model saved to: {model_path}")

preds.as_data_frame().to_csv("h2o_predictions.csv", index=False)
print("Predictions saved to h2o_predictions.csv")

lb = aml.leaderboard
print(lb)