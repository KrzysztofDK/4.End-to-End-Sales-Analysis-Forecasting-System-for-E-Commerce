USE olist;

CREATE TABLE fact_sales AS
SELECT
    oi.order_id,
    o.customer_id,
    oi.product_id,
    oi.seller_id,
    DATE(o.order_purchase_timestamp) AS date,
    oi.price,
    oi.freight_value,
    o.order_status,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date
FROM order_items oi
JOIN orders o
    ON oi.order_id = o.order_id;
