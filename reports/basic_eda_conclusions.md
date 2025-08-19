# Preliminary EDA made in notebook.

## 1. Source Data
- Dataset: Olist Brazilian E-Commerce Dataset,
- Number of tables: 9,
- Number of records: ~100,000.

## 2. Data Issues
- Columns:
  * `geolocation_lat`, `geolocation_lng` in olist_geolocation.csv,
  * `price`, `freight_value`, `shipping_limit_date` in olist_order_items.csv,
  * `payment_value` in olist_order_payments.csv,
  * `review_creation_date`, `review_answer_timestamp` in olist_order_reviews.csv,
  * `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`,
  `order_delivered_customer_date`, `order_estimated_delivery_date` in olist_orders.csv,
  contain numerical values or dates, but were wrote as text.

- Columns names:
  * `order_item_id`, `freight_value` in olist_order_items.csv,
  * `payment_sequential` in olist_order_payments.csv,
  need name conversion to more intelligible.

- Missing values occure in:
   * olist_orders.csv (below 3% of all data),
   * olist_products.csv (12% totaly in four columns).
   * olist_order_reviews.csv (below 3% of all data).

- Duplicate values do not occure.

## 3. Conclusions
- The dataset requires preprocessing (data type fixes and columns names fixes),
- EDA should be performed after resolving the data quality issues.
