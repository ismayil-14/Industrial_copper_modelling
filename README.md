#Copper Industry Sales and Pricing Prediction
Project Overview
The copper industry deals with data related to sales and pricing, which often suffers from skewness and noise. These data issues can lead to inaccurate manual predictions, making it difficult to make optimal pricing decisions. Additionally, capturing and evaluating leads is crucial for the industry, requiring effective classification to predict the likelihood of converting leads into customers.

This project aims to address these challenges using machine learning models for regression and classification. A Streamlit page is also created to allow users to input data and get predictions for selling prices and lead statuses.

Project Goals
Data Exploration and Preprocessing

Explore skewness and outliers in the dataset.
Transform data into a suitable format and perform necessary cleaning and preprocessing steps.
Apply techniques such as data normalization, feature scaling, and outlier detection.
ML Regression Model

Develop a regression model to predict the continuous variable Selling_Price.
ML Classification Model

Develop a classification model to predict lead status (WON or LOST).
Use the STATUS variable, considering WON as success and LOST as failure.
Remove data points with statuses other than WON or LOST.
Streamlit Page

Create an interactive Streamlit page to input data and get predictions for Selling_Price or Status.
Project Steps
1. Data Exploration and Preprocessing
Identify Skewness and Outliers: Analyze the dataset to identify skewed distributions and outliers that may affect model performance.
Data Transformation and Cleaning: Clean the dataset by handling missing values, encoding categorical variables, and normalizing numerical features.
Outlier Detection: Implement techniques to detect and handle outliers to improve model robustness.
2. Regression Model
Model Development: Develop a machine learning regression model using algorithms robust to skewed and noisy data.
Feature Engineering: Create new features and select the most relevant ones to enhance model performance.
Model Evaluation: Evaluate the regression model using metrics such as Mean Squared Error (MSE) and R-squared (R²).
3. Classification Model
Data Preparation: Filter the dataset to include only WON and LOST statuses.
Model Development: Develop a machine learning classification model to predict lead status.
Model Evaluation: Evaluate the classification model using metrics such as accuracy, precision, recall, and F1-score.
4. Streamlit Page
User Interface: Create a user-friendly interface in Streamlit to input feature values.
Prediction Output: Display the predicted Selling_Price or lead Status based on user inputs.
Getting Started
Prerequisites
Python 3.6+
Streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
