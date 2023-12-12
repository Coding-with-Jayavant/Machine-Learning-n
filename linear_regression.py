# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:02:49 2023

@author: jwark
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
data = pd.read_csv(r"C:\Users\jwark\Downloads\archive\weatherHistory.csv")
data.columns
# Select relevant columns
selected_features = ['Humidity', 'Apparent_Temperature_C']

#dropna delete rows who has null values
df = data[selected_features].dropna()
df.head()
# Split the data into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(df[['Humidity']], df['Apparent_Temperature_C'], test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-Square: {r2:.2f}')

# Visualize the simple regression model

sns.scatterplot(x='Humidity', y='Apparent_Temperature_C', data=df)
plt.plot(X_test, y_pred, color='red',label='Linear Regression Model')

plt.show()

