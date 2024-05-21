# Wealth-Distribution-Among-Billionaires-Project

 ### [YouTube Demo](https://youtu.be/NXOUTIyeGuM)

This project analyzes patterns within the wealth distribution of the world's billionaires. Using an annually updated dataset, the analysis aims to uncover contemporary trends and provide insights into the financial landscapes of billionaires across the globe. The objective is to identify patterns, anomalies, and influential factors contributing to wealth disparities, including age, gender, and self-made status.

## Project Overview

- **Objective**: Analyze the wealth distribution among billionaires to discern patterns and influential factors.
- **Dataset**: Annually updated dataset on billionaires.
- **Techniques**: Data preprocessing, exploratory data analysis, feature engineering, and predictive modeling.

## Table of Contents

- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Modeling](#modeling)
- [Conclusion](#conclusion)
- [Bonus Section](#bonus-section-exploring-engineered-features)

## Exploratory Data Analysis (EDA)

### Imports

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

**Explanation**: The code above imports all necessary libraries and modules needed for data visualization, manipulation, and modeling. It includes:
- `matplotlib` and `seaborn` for plotting graphs and visualizations.
- `numpy` and `pandas` for data manipulation and analysis.
- `scikit-learn` for machine learning models and preprocessing.
- `scipy` for statistical operations.
- `warnings` to suppress future warnings during the runtime.

### Loading the Dataset

```python
df = pd.read_csv('Billionaires Statistics Dataset.csv')
```

**Explanation**: This line of code loads the dataset from a CSV file into a pandas DataFrame called `df`.

### Visualizations

#### Pairplot for Numerical Features

```python
colors = df['selfMade'].map({True: 'blue', False: 'red'})
scatter = plt.scatter(df['age'], df['finalWorth'], c=colors, alpha=0.5)
plt.legend(handles=scatter.legend_elements()[0], labels=['Not Self-Made', 'Self-Made'])
plt.title('Pairplot of Numerical Features by Self-Made Status')
plt.xlabel('Age')
plt.ylabel('Final Worth')
plt.show()
```

**Explanation**: This code creates a scatter plot to visualize the relationship between the age of billionaires and their final worth. The points are color-coded based on whether the billionaires are self-made (blue) or not (red).

#### Countplot for Gender

```python
plt.figure()
colors_gender = df['selfMade'].map({True: 'blue', False: 'red'})
plt.hist([df[df['gender'] == 'M']['selfMade'], df[df['gender'] == 'F']['selfMade']],
         stacked=True, color=['blue', 'red'], label=['Male', 'Female'], alpha=0.7)
plt.legend()
plt.title('Count of Self-Made Status by Gender')
plt.xlabel('Self-Made Status')
plt.ylabel('Count')
plt.show()
```

**Explanation**: This code generates a stacked histogram to show the count of self-made and non-self-made billionaires, separated by gender. It helps to understand the distribution of self-made status among male and female billionaires.

## Data Cleaning and Preprocessing

### Handling Missing Values and Outliers

```python
# Create a copy of the DataFrame
df_cleaned = df.copy()

# Handle missing values
columns_to_check = ['gender', 'age', 'selfMade', 'finalWorth']
df_cleaned = df_cleaned.dropna(subset=columns_to_check)

# Anomaly detection and handling
df_cleaned = df_cleaned[(np.abs(zscore(df_cleaned[['age', 'finalWorth']])) < 3).all(axis=1)]
```

**Explanation**:
- The dataset is copied to a new DataFrame `df_cleaned` for cleaning.
- Rows with missing values in critical columns (`gender`, `age`, `selfMade`, and `finalWorth`) are dropped.
- Outliers in the `age` and `finalWorth` columns are removed using z-scores to ensure data integrity.

### Feature Scaling and Engineering

```python
# Feature scaling
scaler = MinMaxScaler()
df_cleaned[['age', 'finalWorth']] = scaler.fit_transform(df_cleaned[['age', 'finalWorth']])

# Binary encoding for gender column
df_cleaned['gender'] = df_cleaned['gender'].apply(lambda x: 1 if x == 'M' else 0)

# Feature Engineering
# 1. Age Categories
bins = [0, 30, 50, 70, 100]
labels = ['Young', 'Adult', 'Senior', 'Centenarian']
df_cleaned['age_category'] = pd.cut(df_cleaned['age'], bins=bins, labels=labels, right=False)

# 2. Normalized Education Enrollment
global_avg_enrollment = df_cleaned['gross_tertiary_education_enrollment'].mean()
df_cleaned['normalized_tertiary_enrollment'] = df_cleaned['gross_tertiary_education_enrollment'] / global_avg_enrollment

# 3. Wealth per Age
df_cleaned['wealth_per_age'] = np.where(df_cleaned['age'] == 0, 0, df_cleaned['finalWorth'] / df_cleaned['age'])

# Save cleaned and engineered DataFrame to a CSV file
save_path = 'cleaned_data.csv'
df_cleaned.to_csv(save_path, index=False)

# Display the cleaned and engineered DataFrame
print(df_cleaned[['age', 'age_category', 'normalized_tertiary_enrollment', 'wealth_per_age', 'gender']].head())
```

**Explanation**:
- `MinMaxScaler` is used to normalize the `age` and `finalWorth` columns.
- The `gender` column is binary encoded (1 for male, 0 for female).
- New features are engineered:
  - `age_category`: Categorizes ages into bins (Young, Adult, Senior, Centenarian).
  - `normalized_tertiary_enrollment`: Normalizes education enrollment based on the global average.
  - `wealth_per_age`: Calculates wealth relative to age.
- The cleaned and engineered DataFrame is saved to a CSV file for future use.

## Modeling

### Preparing the Data

```python
# Drop rows with NaN values and select features and target variable
features = ['age', 'selfMade', 'normalized_tertiary_enrollment', 'wealth_per_age']
target = 'finalWorth'
df_cleaned = df_cleaned.dropna(subset=features + [target])
X = df_cleaned[features]
y = df_cleaned[target]

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Explanation**:
- The dataset is prepared by dropping rows with NaN values in the selected features and target variable (`finalWorth`).
- The features (`age`, `selfMade`, `normalized_tertiary_enrollment`, `wealth_per_age`) and target (`finalWorth`) are separated.
- Missing values in the features are imputed with the mean.
- The data is split into training and testing sets (80% training, 20% testing).
- The features are standardized for better performance of machine learning models.

### Model Training and Evaluation

#### Linear Regression

```python
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)
predictions_lr = model_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, predictions_lr)
print(f'Linear Regression MSE: {mse_lr}')
plot_predictions(y_test, predictions_lr, 'Linear Regression')
```

**Explanation**:
- A Linear Regression model is trained on the standardized training data.
- Predictions are made on the test set.
- The Mean Squared Error (MSE) is calculated to evaluate the model's performance.
- A plot is generated to visualize the predicted vs. actual values.

#### Random Forest

```python
model_rf = RandomForestRegressor()
model_rf.fit(X_train_scaled, y_train)
predictions_rf = model
```

**Explanation**:
- A Random Forest Regressor model is trained on the standardized training data.
- Predictions are made on the test set.
- The Mean Squared Error (MSE) is calculated to evaluate the model's performance.
- A plot is generated to visualize the predicted vs. actual values.

### Additional Models and Evaluation

Similar steps can be followed to train and evaluate other models like Support Vector Regressor (SVR) and Gradient Boosting Regressor.

## Conclusion

The predictive modeling results show the wealth distribution among billionaires based on various demographic and socioeconomic features. The Random Forest and Gradient Boosting models outperform the baseline Linear Regression model, highlighting the importance of data cleaning and feature engineering for accurate predictions. The lower Mean Squared Error (MSE) values for Random Forest and Gradient Boosting models suggest that they capture more complex patterns in the data, providing subtle insights into wealth distribution.

Features like Wealth per Age, normalized tertiary enrollment, and age significantly contribute to predicting billionaires' final worth. This implies that wealth concentration is influenced by a combination of factors such as age, education, and the accumulation of wealth relative to age.

The models and their feature importance analyses provide valuable information about the complexities of wealth distribution among billionaires, informing discussions and policies related to economic inequality and wealth disparities.

## Bonus Section: Exploring Engineered Features

### Scatter Plot for Wealth per Age

```python
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['age'], df_cleaned['wealth_per_age'], alpha=0.5)
plt.title('Scatter Plot of Wealth per Age')
plt.xlabel('Age')
plt.ylabel('Wealth per Age')
plt.show()
```

**Explanation**:
- This scatter plot visualizes the relationship between age and wealth per age.
- It illustrates how wealth varies across different age groups, indicating that individuals with the most wealth are often born into it.

### Box Plot for Normalized Tertiary Enrollment

```python
plt.figure(figsize=(10, 6))
plt.boxplot(df_cleaned['normalized_tertiary_enrollment'], vert=False)
plt.title('Box Plot of Normalized Tertiary Enrollment')
plt.xlabel('Normalized Tertiary Enrollment')
plt.show()
```

**Explanation**:
- This box plot displays the distribution of normalized tertiary education enrollment.
- It helps identify potential outliers or patterns in the normalized tertiary education feature, indicating that most billionaires rise from places with tertiary education enrollment above the global average.

## Contributing

This project provides a comprehensive analysis of wealth distribution among billionaires, utilizing exploratory data analysis, data cleaning, feature engineering, and predictive modeling techniques. The insights gained can contribute to a better understanding of the factors influencing wealth disparities and inform discussions on economic policies and interventions.

For further inquiries or collaboration opportunities, please contact me at hooker7@Purdue.edu.

## Liscencing
This project is licensed under the MIT License. See the LICENSE file for more details.
