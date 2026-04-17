
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
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


# Create the 'results' folder if it doesn't exist
results_dir = Path.cwd() / "results"
results_dir.mkdir(exist_ok=True)

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

# --- 11. SAVE OUTPUTS ---
preds.as_data_frame().to_csv(results_dir / "h2o_predictions.csv", index=False)
print("Predictions saved to h2o_predictions.csv")

model_path = h2o.save_model(model=aml.leader, path=str(results_dir), force=True)
print(f"Model saved to: {model_path}")

lb_df = lb.as_data_frame()
md_table = lb_df.to_markdown(index=False)
with open(results_dir / "h2o_leaderboard.md", "w") as f:
    f.write("# H2O AutoML Leaderboard\n\n")
    f.write(md_table)

# Try to plot variable importance for the leader, or fallback to a base model
try:
    fig = aml.leader.varimp_plot()
    fig.savefig(results_dir / "feature_importance.jpg", format="jpg", dpi=300)
    logger.success("Feature importance plot saved for leader model.")  
    plt.close(fig)
except Exception as e:
    logger.error(f"Leader model does not support variable importance plot: {e}")
    # Fallback: find the best non-ensemble model from the leaderboard
    lb_df = lb.as_data_frame()
    for model_id in lb_df['model_id']:
        model = h2o.get_model(model_id)
        # Only try models that support variable importance
        if hasattr(model, "varimp") and model.varimp() is not None:
            try:
                fig = model.varimp_plot()
                fig.savefig(results_dir / "feature_importance.jpg", format="jpg", dpi=300)
                plt.close(fig)
                logger.success(f"Feature importance plot saved for model: {model_id}")
                break
            except Exception as e:
                logger.error(f"Failed to generate feature importance plot for model {model_id}: {e}")
                continue