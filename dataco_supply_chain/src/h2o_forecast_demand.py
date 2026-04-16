import h2o
from h2o.automl import H2OAutoML

h2o.init()

data_path = "/home/marc/project/supply_chain_DS/dataco_supply_chain/data/processed/demand_forecast_dataset.csv"
df_h2o = h2o.import_file(data_path)

