USE olist;

CREATE TABLE dim_time AS
SELECT DISTINCT
    order_purchase_timestamp AS full_date,
    YEAR(order_purchase_timestamp) AS year,
    MONTH(order_purchase_timestamp) AS month,
    DAY(order_purchase_timestamp) AS day,
    WEEK(order_purchase_timestamp) AS week
FROM orders;
