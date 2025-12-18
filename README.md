# Indian Air Quality Analysis Dashboard

## 📌 Project Overview
This project analyses air quality data across major Indian cities between **2015 and 2019**.  
It combines **exploratory data analysis (EDA)** with an **interactive Streamlit dashboard** to identify pollution patterns and key drivers of air quality.

## 📊 Dataset
- **Source:** Indian Air Quality Dataset  
- **Period:** 2015–2019  
- **Coverage:** 26 Indian cities  
- **Key Variables:** PM2.5, PM10, NO2, CO, SO2, O3, AQI  

## 🧹 Data Cleaning
- Converted the **Date** column to datetime format  
- Addressed missing values in pollutant measurements  
- Removed inconsistent or invalid records to ensure data quality  

## 📈 Exploratory Data Analysis
- Distribution of AQI values  
- Relationships between pollutants and AQI  
- City-wise comparisons and seasonal trends  
- Correlation analysis using a heatmap  

## 🔮 AQI Prediction
A regression-based machine learning model was developed to **predict AQI values** using pollutant concentrations, demonstrating the impact of key pollutants on air quality.

## 🖥️ How to Run the App
```bash
pip install -r requirements.txt
streamlit run air_quality_app.py

## Author
Kitawa Sharon  
MSc Data Science
