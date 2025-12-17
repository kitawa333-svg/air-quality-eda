# Make the Streamlit app file - this creates air_quality_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # For loading the trained model
import numpy as np
import folium
from streamlit_folium import st_folium

# Load cleaned data
df_clean = pd.read_csv("data/cleaned_air_quality.csv")

# Convert Date column to datetime (MANDATORY for .dt and .strftime)
df_clean['Date'] = pd.to_datetime(df_clean['Date'])

# City coordinates (static spatial reference used for mapping)
location_data = pd.DataFrame({
    'lon': [72.5714, 92.7176, 80.6480, 74.8723, 77.5946, 77.4126, 83.9206, 76.7794,
            80.2707, 76.9558, 77.1025, 76.2673, 77.0266, 91.7362, 78.4867, 75.7873,
            86.3833, 76.2673, 88.3639, 80.9462, 72.8777, 85.1376, 91.8933, 85.1836,
            76.9366, 83.2185],
    'lat': [23.0225, 23.7271, 16.5062, 31.6340, 12.9716, 23.2599, 21.8248, 30.7333,
            13.0827, 11.0168, 28.7041, 9.9312, 28.4595, 26.1445, 17.3850, 26.9124,
            23.7167, 9.9312, 22.5726, 26.8467, 19.0760, 25.5941, 25.5788, 20.9495,
            8.5241, 17.6868],
    'location': ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal',
                 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi',
                 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur',
                 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai', 'Patna',
                 'Shillong', 'Talcher', 'Thiruvananthapuram', 'Visakhapatnam']})

st.write(df_clean.columns)


# Load trained model (with error handling)
try:
    model = pickle.load(open("models/model.pkl", "rb"))
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.warning("⚠️ Model file not found. Using placeholder.")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    # Train with dummy data
    X_dummy = np.random.rand(10, 4)
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)

#setting the title of the application
st.title('Indian Air Quality Analysis Dashboard 🌫️')
st.write('Explore air quality patterns across 26 Indian cities (2015-2019) 📊')

# Sidebar navigation
st.sidebar.title('📍 Navigation')
page = st.sidebar.selectbox('Select a page', ['📋 Data Overview', '📊 Exploratory Analysis', '🔮 AQI Prediction'])

# Content for Data Overview page - ONLY ONE BLOCK!
if page == '📋 Data Overview':
    st.header('📋 Dataset Overview')

    # Key metrics
    col1, col2, col3 = st.columns([1, 1, 2])  # 👈 make Date Range column wider
    with col1:
        st.metric("📄 Total Records", len(df_clean))
    with col2:
        st.metric("🏙️ Number of Cities", df_clean['City'].nunique())
    with col3:
         st.metric("📅 Date Range", f"{df_clean['Date'].min().strftime('%Y-%m')} → {df_clean['Date'].max().strftime('%Y-%m')}")

    # Data preview
    st.subheader('👀 Data Preview')
    st.dataframe(df_clean.head(10))

    # Dataset structure
    st.subheader('📐 Dataset Structure')
    st.write(f'**Rows:** {df_clean.shape[0]}, **Columns:** {df_clean.shape[1]}')

    # Column descriptions
    st.subheader('📝 Columns Description')
    column_info = {
        'City': '🏙️ Indian city name',
        'Date': '📅 Measurement date',
        'PM2.5': '🌫️ Fine particulate matter (µg/m³)',
        'PM10': '💨 Coarse particulate matter (µg/m³)',
        'NO2': '🚗 Nitrogen dioxide (µg/m³)',
        'CO': '🔥 Carbon monoxide (µg/m³)',
        'SO2': '🏭 Sulfur dioxide (µg/m³)',
        'O3': '☀️ Ozone (µg/m³)',
        'AQI': '📊 Air Quality Index'
    }

    for col, desc in column_info.items():
        if col in df_clean.columns:
            st.write(f'**{col}:** {desc}')

if page == '📊 Exploratory Analysis':
    st.header('📊 Exploratory Data Analysis')

    # 1. AQI Distribution
    st.subheader('📈 Distribution of AQI Values')
    fig = plt.figure(figsize=(10, 5))
    plt.hist(df_clean['AQI'].dropna(), bins=30, color='hotpink', edgecolor='purple', alpha=0.8)
    plt.xlabel('Air Quality Index (AQI)')
    plt.ylabel('Frequency')
    plt.title('How Often Do Different AQI Levels Occur? 📊')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.caption('📌 Most days fall in "Moderate" to "Unhealthy" range (AQI 100-200)')

    # 2. PM2.5 vs AQI scatter
    st.subheader('🔗 PM2.5 vs AQI Relationship')
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(df_clean['PM2.5'], df_clean['AQI'], color='mediumorchid', alpha=0.4, s=20)
    plt.xlabel('PM2.5 Concentration (µg/m³)')
    plt.ylabel('Air Quality Index (AQI)')
    plt.title('Strong Correlation: More PM2.5 = Poorer Air Quality 📉')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    correlation_val = df_clean[['PM2.5', 'AQI']].corr().iloc[0,1]
    st.caption(f'📊 Correlation: {correlation_val:.2f} (closer to 1 = stronger relationship)')

    # 3. City selection for analysis
    st.subheader('🏙️ City-wise Analysis')
    selected_city = st.selectbox('Select a city to analyze:', df_clean['City'].unique())

    city_data = df_clean[df_clean['City'] == selected_city]

    col1, col2 = st.columns(2)
    with col1:
        avg_aqi = city_data['AQI'].mean()
        st.metric(f"📊 Average AQI in {selected_city}", f"{avg_aqi:.0f}")
    with col2:
        worst_aqi = city_data['AQI'].max()
        worst_day = city_data.loc[city_data['AQI'].idxmax(), 'Date']
        st.metric("🔥 Worst AQI Day", f"{worst_aqi:.0f}", f"on {worst_day.strftime('%d %b %Y')}")

    # 4. Monthly trend for selected city
    st.subheader('📅 Seasonal Pattern')
    fig = plt.figure(figsize=(10, 4))
    city_data['Month'] = city_data['Date'].dt.month
    monthly = city_data.groupby('Month')['AQI'].mean()
    plt.plot(monthly.index, monthly.values, color='mediumvioletred', marker='o', linewidth=2)
    plt.xlabel('Month (1=Jan, 12=Dec)')
    plt.ylabel('Average AQI')
    plt.title(f'Seasonal Pattern in {selected_city} 📈')
    plt.xticks(range(1, 13))
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)
    st.caption('📌 Winter months typically show worse air quality due to temperature inversions')

# 5. Correlation heatmap
    st.subheader('🔥 Correlation Heatmap')

    corr_matrix = df_clean[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'AQI']].corr()

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdPu', center=0)
    plt.title('Pollutant Correlations with AQI 🔗')
    st.pyplot(fig)
    st.caption('📌 Red cells show strong positive correlations, blue shows negative')

    # 6. Spatial distribution map
    st.subheader('🗺️ Spatial Distribution of Cities')

    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

    colors = ['pink', 'lightred', 'purple', 'darkpurple', 'red', 'darkred', 'gray', 'black']

    location_data = df_clean[['City', 'Latitude', 'Longitude']].dropna()
    location_data.columns = ['location', 'lat', 'lon']

    for index, row in location_data.iterrows():
        folium.Marker(
            [row['lat'], row['lon']],
            tooltip=row['location'],
            icon=folium.Icon(color=colors[index % len(colors)])
        ).add_to(m)

    st_folium(m, width=700, height=500)
    st.caption('📍 Interactive map showing the geographic distribution of Indian cities')

    # 7. Top 10 most polluted cities
    st.subheader('🏆 Top 10 Most Polluted Cities')

    city_avg_aqi = (
        df_clean.groupby('City')['AQI']
        .mean()
        .sort_values(ascending=False)
    )

    fig = plt.figure(figsize=(10, 6))
    city_avg_aqi.head(10).plot(
        kind='bar',
        color='mediumorchid',
        edgecolor='purple'
    )
    plt.xlabel('City 🏙️')
    plt.ylabel('Average AQI 📊')
    plt.title('Top 10 Cities by Average AQI 🏆')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)
    st.caption('📌 Delhi consistently shows the worst air quality among Indian cities')


#  predictions page
if page == '🔮 AQI Prediction':
    st.header('🔮 AQI Prediction Tool')
    st.write('Enter pollutant concentrations to predict AQI 🧪')

    col1, col2 = st.columns(2)
    with col1:
        pm25 = st.slider('🌫️ PM2.5 (µg/m³)', 0.0, 500.0, 100.0, help='Fine particulate matter')
        pm10 = st.slider('💨 PM10 (µg/m³)', 0.0, 500.0, 150.0, help='Coarse particulate matter')
    with col2:
        no2 = st.slider('🚗 NO2 (µg/m³)', 0.0, 200.0, 40.0, help='Nitrogen dioxide from vehicles')
        co = st.slider('🔥 CO (µg/m³)', 0.0, 50.0, 2.0, help='Carbon monoxide from combustion')

    # Predict button - ALL CODE GOES INSIDE THIS SINGLE IF BLOCK
    if st.button('🎯 Predict AQI!'):
        prediction = model.predict([[pm25, pm10, no2, co]])[0]
        st.success(f'**🎯 Predicted AQI:** {prediction:.0f}')

        # Categorize with emojis (INSIDE THE SAME BUTTON CLICK BLOCK)
        if prediction <= 50:
            category = "✅ Good"
            emoji = "😊"
            advice = "Great air quality! Perfect for outdoor activities."
        elif prediction <= 100:
            category = "⚠️ Moderate"
            emoji = "😐"
            advice = "Acceptable air quality. Sensitive groups should consider limiting outdoor exertion."
        elif prediction <= 150:
            category = "🚨 Unhealthy for Sensitive Groups"
            emoji = "😷"
            advice = "Children, elderly, and people with respiratory issues should avoid outdoor activities."
        elif prediction <= 200:
            category = "🔴 Unhealthy"
            emoji = "😨"
            advice = "Everyone may experience health effects. Limit outdoor exposure."
        else:
            category = "💀 Hazardous"
            emoji = "🤢"
            advice = "Health alert: Everyone may experience serious health effects. Stay indoors."

        # Show results
        st.info(f'**{emoji} Air Quality Category:** {category}')
        st.write(f'**📋 Health Advice:** {advice}')

# Add footer to sidebar
st.sidebar.markdown('---')
st.sidebar.info('📚 **Data Source:** Indian Air Quality Dataset (2015-2019)')
st.sidebar.info('🛠️ **Built with:** Python, Streamlit, Pandas, Matplotlib')
st.sidebar.info('🎯 **Purpose:** Academic project - Air Quality Analysis')

st.sidebar.success("✅ Model loaded successfully")

