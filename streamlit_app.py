import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
df = pd.read_csv('Indian AQI Data Analysis PRAC1ass.csv')  #update if filename differs

#setting the title of the application
st.title('Air Quality Analysis Dashboard')

#short description
st.write('This dashboard presents exploratory data analysis of air quality data across Indian cities.')

#sidebar navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page', ['Data Overview', 'Exploratory Data Analysis'])

#data overview page
if page == 'Data Overview':

    st.header('Dataset Overview')
    st.write('Preview of the dataset:')
    st.dataframe(df.head())

    st.write('Dataset shape:')
    st.write(df.shape)

    st.write('Column names:')
    st.write(df.columns)

#exploratory data analysis page
if page == 'Exploratory Data Analysis':

    st.header('Exploratory Data Analysis')

    #AQI distribution
    st.subheader('Distribution of AQI')
    fig = plt.figure()
    plt.hist(df['AQI'], bins=30, color='hotpink', edgecolor='black')
    plt.xlabel('Air Quality Index (AQI)')
    plt.ylabel('Frequency')
    plt.title('Distribution of AQI Values')
    st.pyplot(fig)

    #PM2.5 vs AQI scatter plot
    st.subheader('PM2.5 vs AQI')
    fig = plt.figure()
    plt.scatter(df['PM2.5'], df['AQI'], color='purple', alpha=0.5)
    plt.xlabel('PM2.5 Concentration')
    plt.ylabel('Air Quality Index (AQI)')
    plt.title('Relationship Between PM2.5 and AQI')
    st.pyplot(fig)
