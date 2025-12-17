# 🇮🇳 Indian Air Quality Analysis Dashboard

## 📌 Project Overview
This project analyses air quality data across major Indian cities (2015–2019).
It includes exploratory data analysis (EDA) and an interactive Streamlit dashboard.

## 📊 Dataset
- Source: Indian Air Quality Dataset
- Period: 2015–2019
- Cities: 26 Indian cities
- Variables: PM2.5, PM10, NO2, CO, SO2, O3, AQI

## 🧹 Data Cleaning
- Converted Date column to datetime
- Handled missing values
- Removed inconsistent records

## 📈 Exploratory Analysis
- AQI distribution
- Pollutant vs AQI relationships
- City-wise and seasonal trends
- Correlation heatmap

## 🔮 AQI Prediction
A machine learning model predicts AQI based on pollutant levels.

## 🖥️ How to Run the App
```bash
pip install -r requirements.txt
streamlit run air_quality_app.py

## Author
Kitawa Sharon  
MSc Data Science
