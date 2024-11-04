import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# Load the saved pipeline
pipeline = joblib.load('best_model_pipeline.joblib')

def main():
    st.title('Visa Application Status Prediction')
    st.write('Enter the details below to predict visa application status')
    
    # Create input form
    with st.form("visa_prediction_form"):
        # Categorical inputs
        col1, col2 = st.columns(2)
        
        with col1:
            continent = st.selectbox('Continent', 
                ['Asia', 'Europe', 'Africa', 'South America', 
                 'North America', 'Oceania'])
            
            unit_of_wage = st.selectbox('Unit of Wage', 
                ['Year', 'Hour', 'Week', 'Month'])
            
            region_of_employment = st.selectbox('Region of Employment',
                ['Northeast', 'West', 'South', 'Midwest'])
            
            has_job_experience = st.selectbox('Has Job Experience', 
                ['Y', 'N'])
            
            requires_job_training = st.selectbox('Requires Job Training', 
                ['Y', 'N'])
            
        with col2:
            full_time_position = st.selectbox('Full Time Position', 
                ['Y', 'N'])
            
            education_of_employee = st.selectbox('Education of Employee',
                ['High School', "Bachelor's", "Master's", "Doctorate", 
                 'Less than High School'])
            
            # Numerical inputs
            no_of_employees = st.number_input('Number of Employees', 
                min_value=0, value=100)
            
            prevailing_wage = st.number_input('Prevailing Wage ($)', 
                min_value=0.0, value=50000.0)
            
            yr_of_estab = st.number_input('Year of Establishment', 
                min_value=1800, max_value=date.today().year, 
                value=2000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Calculate company age
        company_age = date.today().year - yr_of_estab
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'continent': [continent],
            'unit_of_wage': [unit_of_wage],
            'region_of_employment': [region_of_employment],
            'has_job_experience': [has_job_experience],
            'requires_job_training': [requires_job_training],
            'full_time_position': [full_time_position],
            'education_of_employee': [education_of_employee],
            'no_of_employees': [no_of_employees],
            'prevailing_wage': [prevailing_wage],
            'company_age': [company_age]
        })
        
        # Make prediction
        prediction = pipeline.predict(input_data)
        prediction_proba = pipeline.predict_proba(input_data)
        
        # Display result
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.error('Visa Application Status: Denied')
        else:
            st.success('Visa Application Status: Certified')
            
        # Display probability
        st.write('Prediction Probability:')
        st.write(f'Certified: {prediction_proba[0][0]:.2%}')
        st.write(f'Denied: {prediction_proba[0][1]:.2%}')
        
        # Display input summary
        with st.expander("View Input Summary"):
            st.write(input_data)

if __name__ == '__main__':
    main()