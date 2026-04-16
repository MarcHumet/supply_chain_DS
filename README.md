# supply_chain_DS

## Feature Selection for Monthly Product Demand Forecasting

See the full rationale and details in [feature_selection_demand_forecast.md](feature_selection_demand_forecast.md).

### Objective
Forecast monthly demand (sales or quantity) for each product using historical data. The model predicts future demand per product per month, optionally segmented by market or customer segment.

### Selected Features
The following features are recommended for robust, non-duplicated, and predictive modeling:

```python
required_features = [
	'month',
	'year',
	'Product Card Id',
	'Product Price',
	'Category Name',  # or 'Product Category Id'
	'Department Name',  # or 'Department Id'
	'Market',  # or 'Order Region'
	'Sales',  # target
	# 'Order Item Quantity',  # alternative target
	# 'Shipping Mode',  # optional
	# 'Customer Segment',  # optional
]
```

### Dropped Features
Features dropped include those that are duplicated, too granular, or not available before the forecast period (e.g., Product Name, Product Image, Customer details, transactional IDs, delivery status, etc.).

For more details, see [feature_selection_demand_forecast.md](feature_selection_demand_forecast.md).
