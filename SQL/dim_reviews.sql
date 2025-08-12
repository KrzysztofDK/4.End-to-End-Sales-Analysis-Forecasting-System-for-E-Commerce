USE olist;

CREATE TABLE dim_reviews AS
SELECT DISTINCT
    review_id,
    order_id,
    review_score,
    review_creation_date,
    review_answer_timestamp
FROM order_reviews;
