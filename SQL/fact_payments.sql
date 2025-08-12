USE olist;

CREATE TABLE fact_payments AS
SELECT
    p.order_id,
    p.payment_type,
    p.payment_installments,
    p.payment_value
FROM order_payments p;
