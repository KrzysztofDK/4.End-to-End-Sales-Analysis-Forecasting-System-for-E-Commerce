# Preliminary EDA made in Power BI Conclusions

## 1. Source Data
- Dataset: Olist Brazilian E-Commerce Dataset,
- Number of tables: 9,
- Number of records: ~100,000.

## 2. Data Issues
- Lack of relationships between files,
- Columns:
  * `geolocation_lat`, `geolocation_lng` in olist_geolocation.csv,
  * `price`, `freight_value` in olist_order_items.csv,
  * `payment_value` in olist_order_payments.csv,
  contain numerical values, but were wrote as text.

- Columns names:
  * `order_item_id` in olist_geolocation.csv,
  * `freight_value` in olist_order_items.csv,
  * `payment_sequential` in olist_order_payments.csv,
  need conversion to more intelligible.

- Missing values occure in:
   * olist_orders.csv (below 3% of all data),
   * olist_products.csv (12% totaly in four columns).

- Duplicate values do not occure.

## 3. Data Relationships (Power BI)
- Key relationships:  
  - `orders` 1 --- * `order_reviews`
  - `orders` 1 --- * `order_payments`
  - `orders` 1 --- * `order_customer`
  - `orders` 1 --- * `order_items`
  - `order_items` * --- 1 `products`
  - `order_items` * --- 1 `sellers`
  - `geolocation` * --- * `order_customer`
  - `geolocation` * --- * `sellers`

## 4. Conclusions
- The dataset requires preprocessing (data type fixes and columns names fixes),
- EDA should be performed after resolving the data quality issues.
