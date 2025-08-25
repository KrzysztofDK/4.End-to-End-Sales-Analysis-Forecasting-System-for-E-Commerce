USE olist;

CREATE TABLE dim_customer AS
SELECT DISTINCT
    c.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    g.geolocation_lat,
    g.geolocation_lng
FROM customers c
LEFT JOIN geolocation g
    ON c.customer_zip_code_prefix = g.geolocation_zip_code_prefix;


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


CREATE TABLE dim_products AS
SELECT DISTINCT
    product_id,
    product_category_name,
    product_name_length,
    product_description_length,
    product_photos_qty,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm
FROM products;


CREATE TABLE dim_reviews AS
SELECT DISTINCT
    review_id,
    order_id,
    review_score,
    review_creation_date,
    review_answer_timestamp
FROM order_reviews;


CREATE TABLE dim_sellers AS
SELECT DISTINCT
    seller_id,
    seller_city,
    seller_state
FROM sellers;


CREATE TABLE dim_time AS
SELECT DISTINCT
    order_purchase_timestamp AS full_date,
    YEAR(order_purchase_timestamp) AS year,
    MONTH(order_purchase_timestamp) AS month,
    DAY(order_purchase_timestamp) AS day,
    WEEK(order_purchase_timestamp) AS week
FROM orders;


CREATE TABLE fact_payments AS
SELECT
    p.order_id,
    p.payment_type,
    p.payment_installments,
    p.payment_value
FROM order_payments p;


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