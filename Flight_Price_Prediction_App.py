import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.ensemble import RandomForestRegressor


st.write("""
# Flight Price Prediction App
This app predicts the Price of Flight ticket!
Data obtained from the [Kaggle](https://www.kaggle.com/nikhilmittal/flight-fare-prediction-mh).
""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=10)
Source = st.sidebar.selectbox('Source',('Chennai','Delhi','Kolkata','Mumbai'))
Destination = st.sidebar.selectbox('Destination',('Cochin','Delhi','Hyderabad','Kolkata','New_Delhi'))
Departure_Date = st.sidebar.date_input('Departure_Date', today)
Dep_Time = st.sidebar.slider('Dep_Time', datetime.time(0, 0))
Arrival_Time = st.sidebar.slider('Arrival_Time', datetime.time(0, 0))
Airline_Name_type = st.sidebar.selectbox('Airline_Name_type',('Air India','GoAir','IndiGo','Jet Airways','Jet Airways Business','SpiceJet','Multiple carriers','Multiple carriers Premium economy','Vistara','Vistara Premium economy','Trujet'))
Stops = st.sidebar.selectbox('Stops',('0','1','2','3','4'))

# Encoding date
Day_of_Journey = int(pd.to_datetime(Departure_Date, format="%Y-%m-%dT%H:%M").day)
Month_of_Journey = int(pd.to_datetime(Departure_Date, format="%Y-%m-%dT%H:%M").month)

# Encoding dep_time
Dep_hour = int(Dep_Time.hour)
Dep_min = int(Dep_Time.minute)

# Encoding Arrival_Time
Arrival_hour = int(Arrival_Time.hour)
Arrival_min = int(Arrival_Time.minute)

# Durations
if Arrival_min < Dep_min:
    if Dep_min - Arrival_min > 30:
        Duration_mins = Dep_min + Arrival_min - 30
    else:
        Duration_mins = Arrival_min + Dep_min
    if Arrival_hour < Dep_hour:
        Duration_hours = Arrival_hour - Dep_hour + 23
    else:
        Duration_hours = Arrival_hour - Dep_hour - 1
else:
    Duration_mins = Arrival_min - Dep_min
    if Arrival_hour < Dep_hour:
        Duration_hours = Arrival_hour - Dep_hour + 24
    else:
        Duration_hours = Arrival_hour - Dep_hour

# Encode Stops
Total_Stops = int(Stops)

# name as per columns
Airline = Airline_Name_type

# Load the flight dataset
flight_data = pd.read_csv('data_s.csv')
X = flight_data.drop(['Price'],axis = 1)
y = flight_data['Price']

# Apply OneHotEncoding on Airline, Source and Destination columns

X_Airline = X['Airline']
X_Airline = pd.get_dummies(data=X_Airline, drop_first=True)

X_Source = X['Source']
X_Source = pd.get_dummies(data=X_Source, drop_first=True)

X_Destination = X['Destination']
X_Destination = pd.get_dummies(data=X_Destination, drop_first=True)

X_data = pd.concat([X, X_Airline, X_Source, X_Destination], axis = 1)
X_data.drop(['Airline', 'Source', 'Destination'], axis = 1, inplace = True)

# Create dataframe obtained from input features similar to dataset
data = {'Airline': Airline,
        'Source': Source,
        'Destination': Destination,
        'Total_Stops': Total_Stops,
        'Day_of_Journey': Day_of_Journey,
        'Month_of_Journey': Month_of_Journey,
        'Dep_hour': Dep_hour,
        'Dep_min': Dep_min,
        'Duration_hours': Duration_hours,
        'Duration_mins': Duration_mins,
        'Arrival_hour': Arrival_hour,
        'Arrival_min': Arrival_min}
features = pd.DataFrame(data, index=[0])

# Display features

data_in = {'Source': Source,
           'Destination': Destination,
           'Departure_Date': Departure_Date,
           'Dep_Time': Dep_Time,
           'Arrival_Time': Arrival_Time,
           'Airline_Name_type': Airline_Name_type,
           'Stops': Stops}

features_in = pd.DataFrame(data_in, index=[0])

st.write(features_in)

# Apply OneHotEncoding on Airline, Source and Destination columns on features on features_copy

features_copy = features.copy()
features_copy = pd.get_dummies(features_copy, columns=['Airline','Source','Destination'])

# Create a final dataframe with input data + existing data
Final = pd.concat([features,X])
Final_copy = Final.copy()
Final_copy = pd.get_dummies(Final, columns=['Airline','Source','Destination'])

# Build Regression Model
model = RandomForestRegressor()
model.fit(Final_copy[1:], y)
# Apply Model to Make Prediction
prediction = model.predict(Final_copy[:1])

if st.button('Predict'):
    st.write('''
    ### The Predicted Price of this Flight Ticket is ₹''', round(prediction[0],2))
    # success message
    st.success('This is a success message!')
    st.empty()

# RemoveWARNING: :
st.set_option('deprecation.showPyplotGlobalUse', False)
