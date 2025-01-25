import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load("mlmodel.pkl")
scaler = joblib.load("Scaler.pkl")

st.set_page_config(layout="wide")
st.title("Restaurant Rating Prediction App")
st.caption("This app helps you predict a restaurant review class.")
st.divider()

# Input fields
averagecost = st.number_input("Estimated average cost for two:", min_value=50, max_value=999999, value=1000, step=200)
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])
onlinedelivery = st.selectbox("Restaurant has online booking?", ['Yes', 'No'])
pricerange = st.selectbox("Price range (1 Cheapest, 4 Most Expensive):", [1, 2, 3, 4])
predictbutton = st.button("Predict the reviews!")
st.divider()

bookingstatus = 1 if tablebooking == 'Yes' else 0
deliverystatus = 1 if onlinedelivery == 'Yes' else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    # st.snow()  # Optional snow effect
    try:
        prediction = model.predict(X)
        st.write(prediction)

        if prediction < 2.5:
            st.write("Poor")
        elif prediction < 3.5 :
            st.write("Average")
        elif prediction < 4.0:
            st.write("Good")
        elif prediction < 4.5:
            st.write("Very Good")
        else:
            st.write("Excellent")       

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

