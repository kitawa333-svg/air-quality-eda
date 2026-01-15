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
A **multiple linear regression** model was developed to predict AQI values using key pollutants  
(**PM2.5, PM10, NO2, CO**), demonstrating the strong influence of particulate matter on air quality.

The trained model is saved and reused in the Streamlit dashboard.

## 🖥️ Streamlit Dashboard
The interactive dashboard allows users to:

- Explore the dataset
- View AQI distributions and trends
- Compare pollution levels across cities
- Visualise patterns and relationships
- Interact with the trained prediction model

The app is implemented in:

```bash
streamlit_app.py
pip install -r requirements.txt
streamlit run air_quality_app.py

## Author
Kitawa Sharon  
MSc Data Science
