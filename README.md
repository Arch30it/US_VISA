# US_VISA_Approval_Prediction

The Visa Approval Prediction project is a machine learning application designed to predict the approval status of U.S. visa applications based on historical data. It leverages various applicant and employer attributes, such as job experience, job training requirements, employment type, and prevailing wages, to determine the likelihood of a visa application's success. The project is implemented as a user-friendly Streamlit app, which provides an interactive interface for users to enter applicant details and receive instant predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Model and Pipeline](#model-and-pipeline)

---

## Project Overview
The Visa Approval Prediction app predicts whether a visa application will be approved or denied based on factors such as job experience, education level, job level, region of employment, and company characteristics. The model was trained using a dataset of historical visa applications and uses various feature engineering and preprocessing techniques to improve accuracy.

## Features
- **Input Fields**: The app allows users to input details related to an applicant's job experience, training needs, job level, and employer information.
- **Model Prediction**: Predicts whether the visa application will be approved or denied and provides the probability score.
- **Streamlit UI**: User-friendly interface for making predictions.
- **Machine Learning Pipeline**: Preprocessing steps and model are combined in a single pipeline for consistency and easy deployment.

## Model and Pipeline
The model was trained using the following steps:
1. **Preprocessing**:
   - OneHot Encoding for categorical variables with high cardinality.
   - Ordinal Encoding for ordered categorical variables.
   - Power Transformation for skewed numeric features.
   - Standard Scaling for numerical features.

2. **Pipeline**:
   - All preprocessing steps and the final model are combined in a `Pipeline` for seamless transformation and prediction. The `OneHotEncoder` is configured with `handle_unknown='ignore'` to handle any new categories in production.

3. **Prediction**:
   - The model outputs the predicted visa status (approved or denied) along with the probability score.


   
