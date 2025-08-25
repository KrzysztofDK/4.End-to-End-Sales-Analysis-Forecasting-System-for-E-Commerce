# Conclusions from EDA made in notebook before BERTimbau model creation.

## 1. Source Data
- Dataset: Originally Olist Brazilian E-Commerce Dataset, now sentiment_analysis dataset made in SQL,
- Number of tables: 1,
- Number of records: ~34,000.

## 2. Data Issues
- Data type converting is not required.
- Duplicate values do not occure.

## 3. Conclusions in case of model creation
- The imbalance in distribution means the model may become biased toward positive sentiment,
- The majority of reviews are short (under 50 words). This is typical of e-commerce feedback. For BERT models, short inputs are not a problem, but long ones may need truncation.
