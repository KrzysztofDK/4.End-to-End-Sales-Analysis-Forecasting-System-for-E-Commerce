# Conclusions from EDA made in notebook before Prophet model creation.

## 1. Source Data
- Dataset: Originally Olist Brazilian E-Commerce Dataset, now forecasting dataset made in SQL,
- Number of tables: 1,
- Number of records: ~600.

## 2. Data Issues
- Columns:
  * `order_date` contain date, but is wrote as text.

- There are no missing or null values.

- Duplicate values do not occure.

## 3. Conclusions in case of model creation
- The distribution of daily revenue is slightly right-skewed with a short tail. Good enough for Prophet,
- The earliest data from 2016 appear inconsistent, suggesting incomplete records,
- A significant outlier is observed on November 24th, which could be related to Black Friday, but it is not repeated annually,
- December shows weaker performance after the spike, contrary to typical holiday sales patterns,
- Towards the end of the series, the data appear to abruptly drop, suggesting missing or incomplete data rather than an actual business collapse,
- Monthly revenue highlights the long-term growth trajectory of the platform, with fluctuations becoming more pronounced as the business scales.
