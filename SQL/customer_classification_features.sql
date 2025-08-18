USE olist;


CREATE OR REPLACE VIEW first_order AS
SELECT
	o.customer_id,
    MIN(o.order_approved_at) AS t0_order_approved_at,
    SUBSTRING_INDEX(GROUP_CONCAT(o.order_id ORDER BY o.order_approved_at), ',', 1) AS t0_order_id    
FROM orders o
WHERE o.order_status IN ('delivered', 'invoiced', 'shipped', 'approved')
GROUP BY o.customer_id;


CREATE OR REPLACE VIEW first_order_items AS
SELECT
  f.customer_id,
  SUM(oi.price)          AS total_items_value,
  SUM(oi.freight_value)  AS total_freight_value,
  COUNT(*)               AS n_items,
  COUNT(DISTINCT oi.seller_id) AS n_sellers,
  COUNT(DISTINCT COALESCE(p.product_category_name,'UNKNOWN')) AS n_distinct_categories,
  SUBSTRING_INDEX(
    GROUP_CONCAT(COALESCE(p.product_category_name,'UNKNOWN') ORDER BY oi.price DESC),
    ',', 1
  ) AS top_category_by_priciest_item
FROM first_order f
JOIN order_items oi ON oi.order_id = f.t0_order_id
LEFT JOIN products p ON p.product_id = oi.product_id
GROUP BY f.customer_id;


CREATE OR REPLACE VIEW first_order_payment AS
SELECT
  f.customer_id,
  SUBSTRING_INDEX(
    GROUP_CONCAT(pay.payment_type ORDER BY pay.payment_value DESC, pay.payment_type),
    ',', 1
  ) AS payment_type,
  SUM(pay.payment_value)       AS payment_value,
  MAX(pay.payment_installments) AS payment_installments
FROM first_order f
JOIN order_payments pay ON pay.order_id = f.t0_order_id
GROUP BY f.customer_id;


CREATE OR REPLACE VIEW first_order_customer AS
SELECT
  f.customer_id,
  c.customer_state,
  c.customer_city
FROM first_order f
JOIN customers c ON c.customer_id = f.customer_id;


CREATE OR REPLACE VIEW customer_label AS
SELECT
  f.customer_id,
  CASE WHEN COUNT(o2.order_id) > 0 THEN 1 ELSE 0 END AS y_repeat_90d
FROM first_order f
LEFT JOIN orders o2
  ON o2.customer_id = f.customer_id
 AND o2.order_status IN ('delivered','invoiced','shipped','approved')
 AND o2.order_approved_at >  f.t0_order_approved_at
 AND o2.order_approved_at <= DATE_ADD(f.t0_order_approved_at, INTERVAL 90 DAY)
GROUP BY f.customer_id;


SELECT 'customer_id', 't0_order_approved_at', 't0_order_id'
UNION ALL
SELECT customer_id, t0_order_approved_at, t0_order_id
FROM first_order
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/first_order.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT 'customer_id','total_items_value','total_freight_value','n_items','n_sellers','n_distinct_categories','top_category_by_priciest_item'
UNION ALL
SELECT customer_id,total_items_value,total_freight_value,n_items,n_sellers,n_distinct_categories,top_category_by_priciest_item
FROM first_order_items
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/first_order_items.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT 'customer_id','payment_type','payment_value','payment_installments'
UNION ALL
SELECT customer_id,payment_type,payment_value,payment_installments
FROM first_order_payment
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/first_order_payment.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT 'customer_id','customer_state','customer_city'
UNION ALL
SELECT customer_id,customer_state,customer_city
FROM first_order_customer
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/first_order_customer.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';

SELECT 'customer_id','y_repeat_90d'
UNION ALL
SELECT customer_id,y_repeat_90d
FROM customer_label
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/customer_label.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
