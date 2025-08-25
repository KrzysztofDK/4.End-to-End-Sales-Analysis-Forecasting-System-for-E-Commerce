USE olist;

CREATE TABLE order_reviews_bert (
    review_id VARCHAR(50) PRIMARY KEY,
    order_id VARCHAR(50),
    review_score INT,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date DATETIME,
    review_answer_timestamp DATETIME
);


LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/order_reviews.csv'
IGNORE
INTO TABLE order_reviews_bert
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS
(review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp);


CREATE OR REPLACE VIEW v_sentiment_reviews AS
SELECT
    review_id,
    TRIM(review_comment_message) AS text_pt,
    CASE
        WHEN review_score IN (1,2) THEN 0
        WHEN review_score = 3 THEN 1
        WHEN review_score IN (4,5) THEN 2 
    END AS label
FROM order_reviews_bert
WHERE review_comment_message IS NOT NULL
  AND LENGTH(TRIM(review_comment_message)) >= 20;


SELECT 'review_id','text_pt','label'
UNION ALL
SELECT review_id, text_pt, label
FROM v_sentiment_reviews
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/sentiment_analysis_bertimbau.csv'
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n';
