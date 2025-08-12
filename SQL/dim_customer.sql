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
