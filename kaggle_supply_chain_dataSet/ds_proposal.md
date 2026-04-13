## Plan: Supply Chain Dataset Study & Demand Forecasting

This plan outlines a structured approach to analyze the Fashion and Beauty supply chain dataset, focusing on demand forecasting and safety stock recommendations by SKU. The process follows six ordered phases: data exploration, feature selection, feature engineering, modeling, evaluation metrics, and improvement validation.

---

**Steps**

### Phase 1: Data Exploration & Quality Assessment
1. Load the dataset from kaggle_supply_chain_dataSet/supply_chain_data.csv.
2. Inspect data types, missing values, and unique values per column.
3. Analyze distributions for numerical features (e.g., Price, Number of products sold, Revenue, Stock levels, Lead times, Costs).
4. Explore categorical feature distributions (e.g., Product Type, SKU, Supplier name, Location, Shipping carriers, Transportation modes).
5. Identify outliers and anomalies in key columns (e.g., negative sales, extreme costs, implausible lead times).
6. Summarize data quality issues (missingness, duplicates, inconsistencies).

### Phase 2: Feature Relevance for Demand Forecasting & Safety Stock
7. Review all columns for relevance to demand forecasting and safety stock calculation.
   - Focus on: SKU, Product Type, Number of products sold, Revenue, Stock levels, Lead times, Order quantities, Shipping times, Supplier info, Manufacturing lead time, Defect rates, etc.
8. Exclude columns not directly useful for forecasting (e.g., Customer demographics, if not granular or relevant).
9. Document rationale for inclusion/exclusion of each column.

### Phase 3: Feature Engineering
10. Propose new variables to improve forecasting and safety stock estimation, such as:
    - Sales velocity (products sold per time unit)
    - Stockout frequency
    - Lead time variability
    - Rolling averages (e.g., 7/30-day sales)
    - Demand seasonality indicators
    - Supplier reliability scores
    - Lagged features (previous period sales, stock levels)
    - Categorical encodings (e.g., one-hot for Product Type, Supplier)
11. Specify calculation methods for each engineered feature.

### Phase 4: Modeling Approaches
12. Suggest suitable models for demand forecasting and safety stock:
    - Time series models (ARIMA, SARIMA, Prophet)
    - Machine learning regressors (Random Forest, XGBoost, LightGBM)
    - Deep learning (LSTM, GRU for sequential data)
    - Hybrid approaches (statistical + ML)
13. Recommend model selection based on data size, granularity, and temporal structure.

### Phase 5: Evaluation Metrics
14. Define metrics for model success:
    - Forecasting: MAE, RMSE, MAPE, WAPE
    - Safety stock: Service level, Stockout rate, Inventory turnover
15. Justify metric choices for business relevance.

### Phase 6: Improvement & Model Validity
16. Propose methods to measure improvement:
    - Backtesting with historical data
    - Out-of-sample validation (train/test split, cross-validation)
    - Comparison to baseline (e.g., naive forecast, current safety stock policy)
17. Suggest ongoing monitoring:
    - Drift detection
    - Periodic retraining
    - Business KPIs (e.g., reduction in stockouts, improved service level)

---

**Relevant files**
- kaggle_supply_chain_dataSet/supply_chain_data.csv — primary dataset for analysis
- main.py — for implementing data loading, EDA, feature engineering, modeling, and evaluation

---

**Verification**
1. Confirm data loading and initial summary statistics.
2. Document data quality findings and feature selection rationale.
3. List engineered features and their definitions.
4. Summarize model candidates and justify choices.
5. Specify evaluation metrics and expected business impact.
6. Outline validation and monitoring procedures.

---

**Decisions**
- Focus is on SKU-level demand forecasting and safety stock.
- Exclude features not directly relevant to demand or inventory (unless justified).
- Emphasize explainability and business alignment in feature/model selection.

---

**Further Considerations**
1. If time granularity (daily/weekly/monthly) is not explicit, recommend inferring or requesting clarification.
2. If data is sparse for some SKUs, consider grouping or hierarchical models.
3. If external factors (promotions, holidays) are available, suggest integrating them for improved accuracy.
