import streamlit as st
import pickle
import numpy as np



model , scaler = pickle.load(open('trained_model.sav', 'rb'))


st.title("House Price Prediction App")

st.write("Enter the details below to predict the house price (in $1000s).")


size = st.number_input("House size (m²):", min_value=20, max_value=1000, value=100)
bedrooms = st.number_input("Number of bedrooms:", min_value=1, max_value=10, value=2)


features = np.array([[size, bedrooms]])

features_scaled = scaler.transform(features)

if st.button("Predict Price"):
    prediction = model.predict(features_scaled)
    st.success(f"Estimated Price: {prediction[0]:,.2f}k")
    
    

