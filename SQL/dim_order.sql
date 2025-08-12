USE olist;

CREATE TABLE dim_order AS
SELECT DISTINCT
  order_id,
  customer_id,
  DATE(order_purchase_timestamp) AS order_purchase_date,
  order_status,
  order_approved_at,
  order_delivered_carrier_date,
  order_delivered_customer_date,
  order_estimated_delivery_date
FROM orders;
