# Copper Industry Sales and Pricing Prediction

## Project Overview
The copper industry deals with data related to sales and pricing, which often suffers from skewness and noise. These data issues can lead to inaccurate manual predictions, making it difficult to make optimal pricing decisions. Additionally, capturing and evaluating leads is crucial for the industry, requiring effective classification to predict the likelihood of converting leads into customers.

This project aims to address these challenges using machine learning models for regression and classification. A Streamlit page is also created to allow users to input data and get predictions for selling prices and lead statuses.

## Project Goals
### 1.Data Exploration and Preprocessing

* Explore skewness and outliers in the dataset.
* Transform data into a suitable format and perform necessary cleaning and preprocessing steps.
* Apply techniques such as data normalization, feature scaling, and outlier detection.
### 2.ML Regression Model

* Develop a regression model to predict the continuous variable Selling_Price.
### 3.ML Classification Model

* Develop a classification model to predict lead status (WON or LOST).
* Use the STATUS variable, considering WON as success and LOST as failure.
* Remove data points with statuses other than WON or LOST.


## Prerequisites
- Python 3.7
- Streamlit
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
