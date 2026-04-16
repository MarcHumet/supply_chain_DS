# Feature Selection for Monthly Product Demand Forecasting

## Objective
Forecast monthly demand (sales or quantity) for each product in the supply chain using historical data. The model should be able to predict future demand per product per month, optionally segmented by market or customer segment.

## Feature Analysis

### Essential Features to Keep
- **month, year**: For time grouping and trend/seasonality modeling.
- **Product Card Id**: Unique product identifier (use for grouping and as a key feature).
- **Product Price**: Price of the product (if available before the forecast period).
- **Category Name** (or Product Category Id): Product category (keep only one).
- **Department Name** (or Department Id): Product department (keep only one).
- **Market** (or Order Region): Regional information (keep only one).
- **Sales**: Target variable for regression (total sales per product per month).
- **Order Item Quantity**: Alternative target (units sold per product per month).
- **Shipping Mode**: Optional, if it affects demand.
- **Customer Segment**: Optional, if you want to segment forecasts by customer type.

### Features to Discard
- **Product Name, Product Image**: Duplicated by Product Card Id.
- **Category Id**: If Category Name is kept.
- **Department Id**: If Department Name is kept.
- **Customer Fname, Customer Lname, Customer Id, Customer Street, Customer Zipcode, Customer City, Customer State, Customer Country**: Too granular for monthly product-level forecast; drop all except maybe Customer Segment.
- **Order Id, Order Item Id, Order Item Cardprod Id**: Transactional, not needed for monthly aggregation.
- **Order Customer Id**: Duplicated with Customer Id.
- **Order City, Order Country, Order State, Order Zipcode**: Duplicated with Market/Region; keep only one regional feature.
- **Order date (DateOrders), shipping date (DateOrders)**: Use only for extracting month/year; after that, drop.
- **Latitude, Longitude**: Too granular for monthly product-level forecast.
- **Order Profit Per Order, Order Item Profit Ratio, Order Item Discount, Order Item Discount Rate, Order Item Product Price, Order Item Total, Benefit per order, Sales per customer**: Many are derived from Sales/Quantity/Price; keep only if modeling price sensitivity or promotions.
- **Late_delivery_risk, Delivery Status, Days for shipping (real), Days for shipment (scheduled)**: Not known before the month, so not useful for forecasting.

## Recommended Feature List (Pythonic)

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

## Notes
- For monthly forecasting, aggregate your data by `[Product Card Id, month, year, Market, ...]` as needed.
- If you want to forecast by region or segment, keep those features; otherwise, drop.
- Drop all features that are only available after the fact (e.g., delivery status, actual shipping days).

---

This feature selection ensures your model is focused, avoids data leakage, and is ready for robust monthly demand forecasting per product.
