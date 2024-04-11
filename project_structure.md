# Data Science Project Structure

This document provides a guideline to a standard Data Science Project with the most common steps explained. We assume the objective of the project is taken and the data is collected. We will focus on data cleaning, feature engineering, model training, hyperparameter tuning, model evaluating and interpretation.

Also, this guide is focusing on supervised machine learning problems, but most part of it is applicable on unsupervised problems.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is a crucial initial step in any data science project. It involves analyzing and summarizing the main characteristics of a dataset, often using statistical graphics and other data visualization methods.

EDA can be done after data cleaning but we prefer apply it first to understand the data we will work. Then, while doing data cleaning we can do some EDA to analyze the changes in the data.

### Understanding the Data

Begin by obtaining the dataset and understanding its structure. Identify the format and the size (number of rows and columns). Review the data description or metadata to understand what each column represents.

### Statistical Summary

Calculate basic summary statistics such as mean, median, mode, minimum, maximum, and standard deviation for numerical columns. For categorical variables, determine frequency counts.

### Univariate Analysis

Analyze individual variables to understand their distribution and characteristics. For numerical variables, use histograms, box plots, or density plots to visualize the distribution. For categorical variables, create bar charts or pie charts to show the frequency of each category.

### Bivariate and Multivariate Analysis

Explore relationships between variables. Use scatter plots, pair plots (for smaller datasets), or correlation matrices to identify patterns and dependencies between numerical variables. For categorical vs. numerical variable relationships, box plots or violin plots can be useful. Analyze correlations between variables using correlation coefficients like Pearson, Spearman, or Kendall.

## Data Cleaning and Preprocessing

Data cleaning and preprocessing are crucial steps in any Data Science project to ensure that the data is in a suitable format for analysis and modeling. This step involves handling missing or erroneous data, transforming data into a usable format, and preparing it for further analysis.

It's important to mention that all the transformations done may have to be replicated in the test data and also in production.

### Handling Missing Data

Missing data can arise due to various reasons such as data entry errors, equipment malfunction, or intentional omission. It's important to address missing values appropriately to avoid biased or inaccurate analyses. Also, it's interesting to understand why there is missing data because the absence of data could reveal information and hide patterns. Missing data could be completely random (MCAR), at random (MAR) or not at random (MNAR) where the third one is the more relevant to study carefully.

The techniques usually used to handle missing data are:

1. **Deletion**. We can delete entire rows and columns to solve the problem. This is the most easy way but we lost information and, if it's MNAR, valuable data.

2. **Imputation**. To avoid delete rows we can give them a neutral value which it doesn't bias the analysis and the model. This can be reach with mean, median and mode that are statistics to measure the center of a distribution. In time series data we can impute with the last observed value or the next observed value.

3. **Prediction**. It's a more advanced way to impute values where you train a model like KNN a predict the values for the missings. This method can be more accurate but requires more computational resources and you have to deploy it next to the main model to impute the missings values in production.

There are other advanced techniques like interpolation, matrix factorization or deep learning methods like autoencoders.

### Dealing with Outliers

To ensure that the presence of extreme values does not unduly influence the analysis or modeling process. Outliers are data points that significantly differ from other observations in a dataset and can arise due to various reasons such as measurement errors, natural variation, or rare events. The same carefull mind must be apply here as in missing data.

1. **Removing**. Consider removing outliers if they are likely to be due to data entry errors or do not represent genuine observations. Use domain knowledge to determine a threshold beyond which data points are considered outliers.

2. **Transforming data**. Apply transformations to the data to reduce the impact of outliers without removing them entirely. Common transformations include logarithmic, square root, or Box-Cox transformations.

3. **Winsorization**. Involves replacing extreme outlier values with less extreme values (e.g., replacing values above the 95th percentile with the 95th percentile value).

In practice, removing outliers leads a worst model performace so it isn't an advisable approach unless the outliers are really error data. This is because in test dataset and production data would have outliers.

### Handling Inconsistent Data

Handling inconsistent data involves identifying and rectifying discrepancies or anomalies in the dataset to ensure its reliability and suitability for analysis. Inconsistent data can arise due to various reasons such as human error during data entry, differences in data formats or units, or inconsistencies in data sources.

1. **Standardizing Data Formats**. Ensure consistency in data formats across the dataset. Convert data into a uniform format (e.g., date formatting, numerical representation) to facilitate analysis. Address inconsistencies in text fields (e.g., capitalization, spelling variations) by standardizing text data using text preprocessing techniques.

2. **Resolving Data Entry Errors or Conflicting Information**. Identify and correct data entry mistakes (e.g., typographical errors, duplicate entries) by reviewing and cleaning data records. Resolve conflicts or discrepancies between different data sources or variables.

3. **Handling Duplicates**. Detect and remove duplicate records or entries to avoid redundancy and ensure data integrity.

4. **Dealing with Inconsistent Units or Scales**. Address differences in measurement units or scales within the dataset.

### Feature Engineering

Feature engineering involves creating new features or transforming existing features to make machine learning algorithms work more effectively. Good feature engineering can lead to improved model performance and better insights from the data. It can handle complex data structures and reduce overfitting.

#### Encoding Categorical Variables

Convert categorical variables into numerical representations using techniques like one-hot encoding or label encoding. One-hot encoding is the prefered method over label encoding because the second one may introduce bias based on distance concept. But one-hot enconding can introduce so many new variables and lead to a dimensionality curse problem caused by high cardinality variables. More advanced techniques to address that is to create buckets or use embeddings.

#### Normalization/Scaling

Scale numerical features to a similar range (e.g., using Min-Max scaling or Z-score normalization) to avoid bias in model training caused by diferences in variables magnitudes.

#### Handling Text Data

Text vectorization (Bag-of-Words, TF-IDF) and word embeddings can help to treat text variables.

#### Handling Date and Time Data

Extract relevant components (year, month, day, hour, minute) from date/time data to capture seasonal patterns or time-dependent trends.

#### Creating new variables

Sometimes can be helpfull to create new variables from the existing ones. This could help the model to better capture the intrinsic relationships between the variables. 

### Handling Imbalanced Data

Class imbalance issues happen when the objetive variable has classes that are underrepresenting, this is, when there is less data of one or more classes than the others. This can lead a poor performance of the model because it tends more to tag the data as the majority classes or to not learn enough from the minority classes.

1. **Oversampling**. Increase the number of instances in the minority class by randomly duplicating samples. One way is generating synthetic data with SMOTE (Synthetic Minority Over-sampling Technique) technique based on interpolation between existing minority class instances.

2. **Undersampling**. Reduce the number of instances in the majority class by randomly removing samples.

3. **Class Weight Adjustment**. Modify class weights in classifiers (e.g., logistic regression, SVM) to give higher importance to minority class instances during training.

On the evaluation step it's important to use appropriate metrics like precision, recall and F1-score. If you use cross-validation you have to perform stratified cross-validation instead.

### Feature Importance

Usually it could be interesting to know the relevance of each variable when predecting the outcome. The relationship between dependent and independent variables can be calculated with Pearson's correlation coefficient as we describe in EDA section. But another metric we can analyze is the feature importance. We can train a simple random forest model and it natively calculates this metric. Variables with a very low feature importance in small datasets (few features) or simply a low feature importance in large datasets can be deleted from the modeling because we can assume they are only noise.

In practice, it's not an advisable approach to remove low feature importace or low correlation variables because, at the end of the day, they give some helpfull information (mostly to complex algorithms). 

## Modeling

Work in progress!

## Hyperparameter Tuning

Work in progress!

## Model Evaluation

Work in progress!

## Interpretation and Visualization

Work in progress!
