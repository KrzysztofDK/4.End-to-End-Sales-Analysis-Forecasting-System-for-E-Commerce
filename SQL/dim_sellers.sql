USE olist;

CREATE TABLE dim_sellers AS
SELECT DISTINCT
    seller_id,
    seller_city,
    seller_state
FROM sellers;
