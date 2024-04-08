# Data Science Project Structure

This document provides a guideline to a standard Data Science Project with the most common steps explained. We assume the objective of the project is taken and the data is collected. We will focus on data cleaning, feature engineering, model training, hyperparameter tuning, model evaluating and interpretation.

Also, this guide is focusing on supervised machine learning problems, but most part of it is applicable on unsupervised problems.

## Exploratory Data Analysis (EDA)

EDA can be done after data cleaning but we prefer

## Data Cleaning and Preprocessing

Data cleaning and preprocessing are crucial steps in any Data Science project to ensure that the data is in a suitable format for analysis and modeling. This step involves handling missing or erroneous data, transforming data into a usable format, and preparing it for further analysis.

### Handling Missing Data

Missing data can arise due to various reasons such as data entry errors, equipment malfunction, or intentional omission. It's important to address missing values appropriately to avoid biased or inaccurate analyses. Also, it's interesting to understand why there is missing data because the absence of data could reveal information and hide patterns. Missing data could be completely random (MCAR), at random (MAR) or not at random (MNAR) where the third one is the more relevant to study carefully.

The techniques usually used to handle missing data are:

1. **Deletion**. We can delete entire rows and columns to solve the problem. This is the most easy way but we lost information and, if it's MNAR, valuable data.

2. **Imputation**. To avoid delete rows we can give them a neutral value which it doesn't bias the analysis and the model. This can be reach with mean, median and mode that are statistics to measure the center of a distribution. In time series data we can impute with the last observed value or the next observed value.

3. **Prediction**. It's a more advanced way to impute values where you train a model like KNN a predict the values for the missings. This method can be more accurate but requires more computational resources and you have to deploy it next to the main model to impute the missings values in production.

There are other advanced techniques like interpolation, matrix factorization or deep learning methods like autoencoders.

## Feature Engineering

Work in progress!

## Modeling

Work in progress!

## Hyperparameter Tuning

Work in progress!

## Model Evaluation

Work in progress!

## Interpretation and Visualization

Work in progress!
