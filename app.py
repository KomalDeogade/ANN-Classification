import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model and encoders
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:  
    label_encoder = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)


# Define the Streamlit app
st.title('Customer Churn Prediction')
# Create input fields for user to enter customer data
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age',18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

# One-hot encode the 'Geography' feature
geography_encoded = onehot_encoder.transform(pd.DataFrame([[geography]], columns=['Geography'])).toarray()
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded 'Geography' with the rest of the input data
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

# Reorder features to match the scaler training order
input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# predict churn probability
prediction = model.predict(input_data_scaled)
churn_probability = prediction[0][0]

if prediction >  0.5:
    st.write(f'Churn Probability: {churn_probability:.2f} - Likely to Churn')
else:
    st.write(f'Churn Probability: {churn_probability:.2f} - Unlikely to Churn')
    