SET GLOBAL local_infile = 1;
USE olist;

CREATE TABLE customers (
    customer_id VARCHAR(255),
    customer_unique_id VARCHAR(255),
    customer_zip_code_prefix INT,
    customer_city VARCHAR(255),
    customer_state VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/customers.csv'
INTO TABLE customers
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

CREATE TABLE geolocation (
    geolocation_zip_code_prefix INT,
    geolocation_lat VARCHAR(255),
    geolocation_lng VARCHAR(255),
    geolocation_city VARCHAR(255),
    geolocation_state VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/geolocation.csv'
INTO TABLE geolocation
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

UPDATE geolocation
SET geolocation_lat = REPLACE(geolocation_lat, ',', '.'),
    geolocation_lng = REPLACE(geolocation_lng, ',', '.');

ALTER TABLE geolocation
MODIFY geolocation_lat DECIMAL(33,30),
MODIFY geolocation_lng DECIMAL(33,30);


CREATE TABLE order_items (
    order_id VARCHAR(255),
    order_item_id INT,
    product_id VARCHAR(255),
    seller_id VARCHAR(255),
    shipping_limit_date DATETIME,
    price VARCHAR(255),
    freight_value VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/order_items.csv'
INTO TABLE order_items
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

UPDATE order_items
SET price = REPLACE(price, ',', '.'),
    freight_value = REPLACE(freight_value, ',', '.');

ALTER TABLE order_items
MODIFY price DECIMAL(10,2),
MODIFY freight_value DECIMAL(10,2);

CREATE TABLE order_payments (
    order_id VARCHAR(255),
    payment_sequential INT,
    payment_type VARCHAR(255),
    payment_installments INT,
    payment_value VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/order_payments.csv'
INTO TABLE order_payments
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

UPDATE order_payments
SET payment_value = REPLACE(payment_value, ',', '.');

ALTER TABLE order_payments
MODIFY payment_value DECIMAL(10,2);

CREATE TABLE order_reviews (
    review_id VARCHAR(255),
    order_id VARCHAR(255),
    review_score INT,
    review_creation_date DATETIME,
    review_answer_timestamp DATETIME
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/order_reviews_sql.csv'
INTO TABLE order_reviews
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(review_id, order_id, review_score, @rcd, @rat)
SET
    review_creation_date = NULLIF(@rcd,''),
    review_answer_timestamp = NULLIF(@rat,'');

CREATE TABLE orders (
    order_id VARCHAR(255),
    customer_id VARCHAR(255),
    order_status VARCHAR(255),
    order_purchase_timestamp DATETIME,
    order_approved_at DATETIME,
    order_delivered_carrier_date DATETIME,
    order_delivered_customer_date DATETIME,
    order_estimated_delivery_date DATETIME
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/orders.csv'
INTO TABLE orders
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(order_id, customer_id, order_status,
 @opt, @oaa, @odc, @odcc, @oedd)
SET
    order_purchase_timestamp = NULLIF(@opt,''),
    order_approved_at = NULLIF(@oaa,''),
    order_delivered_carrier_date = NULLIF(@odc,''),
    order_delivered_customer_date = NULLIF(@odcc,''),
    order_estimated_delivery_date = NULLIF(@oedd,'');

CREATE TABLE products (
    product_id VARCHAR(255),
    product_category_name VARCHAR(255),
    product_name_lenght VARCHAR(255),
    product_description_lenght VARCHAR(255),
    product_photos_qty VARCHAR(255),
    product_weight_g VARCHAR(255),
    product_length_cm VARCHAR(255),
    product_height_cm VARCHAR(255),
    product_width_cm VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/products.csv'
INTO TABLE products
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

UPDATE products
SET product_name_lenght = NULLIF(TRIM(REPLACE(product_name_lenght, ',', '.')), ''),
    product_description_lenght = NULLIF(TRIM(REPLACE(product_description_lenght, ',', '.')), ''),
    product_photos_qty = NULLIF(TRIM(REPLACE(product_photos_qty, ',', '.')), ''),
    product_weight_g = NULLIF(TRIM(REPLACE(product_weight_g, ',', '.')), ''),
    product_length_cm = NULLIF(TRIM(REPLACE(product_length_cm, ',', '.')), ''),
    product_height_cm = NULLIF(TRIM(REPLACE(product_height_cm, ',', '.')), ''),
    product_width_cm = NULLIF(TRIM(REPLACE(product_width_cm, ',', '.')), '');

ALTER TABLE products
MODIFY product_name_lenght DECIMAL(8,2),
MODIFY product_description_lenght DECIMAL(8,2),
MODIFY product_photos_qty DECIMAL(8,2),
MODIFY product_weight_g DECIMAL(8,2),
MODIFY product_length_cm DECIMAL(8,2),
MODIFY product_height_cm DECIMAL(8,2),
MODIFY product_width_cm DECIMAL(8,2);

CREATE TABLE sellers (
    seller_id VARCHAR(255),
    seller_zip_code_prefix INT,
    seller_city VARCHAR(255),
    seller_state VARCHAR(255)
);

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/sellers.csv'
INTO TABLE sellers
FIELDS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;