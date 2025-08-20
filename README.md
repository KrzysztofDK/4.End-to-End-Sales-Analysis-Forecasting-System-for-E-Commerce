# 📊 4.End-to-End Smart Sales Analysis & Forecasting System for E-Commerce
Stack: Python, Pandas, SQL, ML, ANN/LSTM, Power BI, Streamlit, GitHub, Docker.

## 🧠 About the Project
The project's goal is to create an intelligent sales analysis and forecasting system for an e-commerce platform that allows for the analysis of customer data, product ratings, and sales trends. The system provides a user interface for prediction, customer classification, and sentiment analysis based on transaction data and reviews.

## 📁 Dataset
Dataset taken from Kaggle -> https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

This is a Brazilian ecommerce public dataset of orders made at Olist Store. The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. We also released a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.

This is real commercial data, it has been anonymised, and references to the companies and partners in the review text have been replaced with the names of Game of Thrones great houses.

# Context of Dataset
This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. See more on our website: www.olist.com

After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

Data Schema avaible in artifacts/images folder as data_schema.png.

## ⚙️ Technologies Used (in progress)
- Python 3.10
- Jupyter Notebook
- VS Code
- SQL
- Power BI
- Pandas
- Numpy
- Matplotlib
- Seaborn
- scikit-learn
- Streamlit
- pyyaml
- mysql-connector-python
- python-dotenv
- tensorflow
- openpyxl
- imblearn

## 🧪 Steps Performed
1. **Exploratory Data Analysis**
- EDA avaible in the notebook/raw_data_EDA.ipynb file
- conclusions avaible in the reports/basic_eda_conclusions.md file
   
2. **Data Cleaning**
- parameters for cleaning avaible in the configs/data_cleaning_config.yaml file
   
3. **Preparing data for the dashboard using SQL**
- avaible in the SQL/ location
   
4. **Preparing a dashboard using Power BI (Star Schema)**
- avaible in the reports/dashboard.pbix or dashboard.pdf files

5. **Customer classification (ANN Keras  
XGBoost will probably give a better result, but the project is for demonstration purposes only)** (in progress)
- Whether a new customer will make a second purchase within 90 days of their first transaction or not (binary).
- SQL code avaible in SQL/customer_classification_features.sql and data_ingestion.py
- EDA avaible in the notebook/classification_EDA.ipynb file

6. **Sentiment analysis**

7. **Sale forecasting**

8. **Streamlit: A Simple Prediction App**

🧑‍💼 Author: Krzysztof Kopytowski
📎 LinkedIn: https://www.linkedin.com/in/krzysztof-kopytowski-74964516a/
📎 GitHub: https://github.com/KrzysztofDK