USE olist;


CREATE OR REPLACE VIEW v_orders AS
SELECT
    o.order_id,
    o.customer_id,
    DATE(o.order_purchase_date) AS order_date,
    o.order_status
FROM dim_order o;


CREATE OR REPLACE VIEW v_payments AS
SELECT
    p.order_id,
    SUM(p.payment_value) AS order_payment_value
FROM fact_payments p
GROUP BY p.order_id;


CREATE OR REPLACE VIEW v_customers AS
SELECT
    c.customer_id,
    c.customer_unique_id,
    c.customer_state,
    c.customer_city
FROM dim_customer c;


CREATE OR REPLACE VIEW v_order_revenue AS
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    p.order_payment_value
FROM v_orders o
JOIN v_payments p ON o.order_id = p.order_id;


CREATE OR REPLACE VIEW v_daily_revenue AS
SELECT
    o.order_date,
    SUM(o.order_payment_value) AS daily_revenue
FROM v_order_revenue o
GROUP BY o.order_date
ORDER BY o.order_date;


SELECT 'order_date','daily_revenue'
UNION ALL
SELECT order_date, daily_revenue
FROM v_daily_revenue
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/forecasting_prophet_daily_revenue.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n';
