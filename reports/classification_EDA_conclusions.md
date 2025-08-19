# Conclusions from EDA made in notebook before ANN model creation.

## 1. Source Data
- Dataset: Originally Olist Brazilian E-Commerce Dataset, now customer_classification dataset made in SQL,
- Number of tables: 1,
- Number of records: ~100,000.

## 2. Data Issues
- Columns:
  * `t0_order_date`, `payment_installments` contain numerical values or dates, but were wrote as text or float.

- Missing values occure in:
   * `payment_type`, `payment_value`, `payment_installments` columns.

- Duplicate values do not occure.

## 3. Conclusions in case of model creation
- The dataset requires preprocessing (data type fixes and missing data fixes),
- Target classes are very unbalanced. Weighting is required,
- Highly skewed distributions in histograms of numerical features. Scaling is required,
- Visible cyclic categorical data. Possible encoding to sin/cos,
- payment_installments I will leave as integer, but payment_type I will encoder with OneHotEncoder,
- I will encode categories with OneHotEncoder or target encoding (maybe top 20 and rest dump to "others"),
- I will encode customer_state, but customer_city has too many values and I will drop it,
- total_items_value, payment_value and first_gmv have too strong coorelation. I will leave only one; payment_value.
