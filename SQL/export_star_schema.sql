USE olist;

SELECT 'customer_id', 'customer_unique_id', 'customer_city', 'customer_state', 'geolocation_lat', 'geolocation_lng'
UNION ALL
SELECT customer_id, customer_unique_id, customer_city, customer_state, geolocation_lat, geolocation_lng
FROM dim_customer
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_customers.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'seller_id', 'seller_city', 'seller_state'
UNION ALL
SELECT seller_id, seller_city, seller_state
FROM dim_sellers
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_sellers.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'product_id', 'product_category_name', 'product_name_length', 'product_description_length', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm'
UNION ALL
SELECT product_id, product_category_name, product_name_length, product_description_length, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm
FROM dim_products
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_products.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'full_date', 'year', 'month', 'day', 'week'
UNION ALL
SELECT full_date, year, month, day, week
FROM dim_time
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_time.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'review_id', 'order_id', 'review_score', 'review_creation_date', 'review_answer_timestamp'
UNION ALL
SELECT review_id, order_id, review_score, review_creation_date, review_answer_timestamp
FROM dim_reviews
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_reviews.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'order_id', 'customer_id', 'product_id', 'seller_id', 'date', 'price', 'freight_value', 'order_status', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'
UNION ALL
SELECT order_id, customer_id, product_id, seller_id, date, price, freight_value, order_status, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
FROM fact_sales
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/fact_sales.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'order_id', 'payment_type', 'payment_installments', 'payment_value'
UNION ALL
SELECT order_id, payment_type, payment_installments, payment_value
FROM fact_payments
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/fact_payments.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';

SELECT 'order_id', 'customer_id', 'order_purchase_date', 'order_status', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'
UNION ALL
SELECT order_id, customer_id, order_purchase_date, order_status, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
FROM dim_order
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_order.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\n';