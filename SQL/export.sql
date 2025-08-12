USE olist;

SELECT * FROM dim_customer
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_customers.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';

SELECT * FROM dim_sellers
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_sellers.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';

SELECT * FROM dim_products
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_products.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';

SELECT * FROM dim_time
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_time.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';

SELECT * FROM dim_reviews
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/dim_reviews.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';

SELECT * FROM fact_sales
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/fact_sales.csv'
FIELDS TERMINATED BY ';'
ENCLOSED BY ''
LINES TERMINATED BY '\n';