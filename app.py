import streamlit as st
import pickle
import numpy as np

# Load the model
with open('lasso.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title("PM10 Predictor")

# Year
year = st.selectbox('Year', [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

# Month
month = st.selectbox('Month', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Day
day = st.selectbox('Day', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])

# Hour
hour = st.selectbox('Hour', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

# Input fields for other features
pm25 = st.number_input('PM2.5')
so2 = st.number_input('SO2')
no2 = st.number_input('NO2')
co = st.number_input('CO')
o3 = st.number_input('O3')
temperature = st.number_input('Temperature')
pressure = st.number_input('Pressure')
dew_point = st.number_input('Dew Point')
rain = st.number_input('Rain')

# Wind direction variables
wd_ENE = st.number_input('wd_ENE')
wd_ESE = st.number_input('wd_ESE')
wd_N = st.number_input('wd_N')
wd_NE = st.number_input('wd_NE')
wd_NNE = st.number_input('wd_NNE')
wd_NNW = st.number_input('wd_NNW')
wd_NW = st.number_input('wd_NW')
wd_S = st.number_input('wd_S')
wd_SE = st.number_input('wd_SE')
wd_SSE = st.number_input('wd_SSE')
wd_SSW = st.number_input('wd_SSW')
wd_SW = st.number_input('wd_SW')
wd_W = st.number_input('wd_W')
wd_WNW = st.number_input('wd_WNW')
wd_WSW = st.number_input('wd_WSW')

if st.button('Predict PM10'):
    # Create the query array including all input features
    query = np.array([
        year, month, day, hour, pm25, so2, no2, co, o3, temperature, pressure, dew_point, rain, 
        wd_ENE, wd_ESE, wd_N, wd_NE, wd_NNE, wd_NNW, wd_NW, wd_S, wd_SE, wd_SSE, wd_SSW, wd_SW, wd_W, wd_WNW, wd_WSW
    ])
    query = query.reshape(1, -1)
    
    # Make prediction and display result
    predicted_pm10 = model.predict(query)
    st.title("The predicted PM10 level is " + str(round(predicted_pm10[0], 2)))
