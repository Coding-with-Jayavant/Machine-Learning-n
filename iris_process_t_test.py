# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:16:51 2023

@author: jwark
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_iris

# Download the Iris dataset
iris = load_iris()
iris
#here we concatinate the data
data = np.c_[iris.data, iris.target]

columns = np.append(iris.feature_names, ["target"])

df = pd.DataFrame(data, columns=columns)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Data visualization
# Pairplot for visualizing relationships between variables
sns.pairplot(df, hue="target", palette="viridis")
plt.title("Pairplot of Iris Dataset")
plt.show()

# Boxplot for each feature
sns.boxplot(x="target", y="sepal length (cm)", data=df)


# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")


# Perform a t-test between two species for sepal length
setosa_sepal_length = df[df["target"] == 0]["sepal length (cm)"]
versicolor_sepal_length = df[df["target"] == 1]["sepal length (cm)"]

#t stat->comapre the means of two groups
#p_value-> calculate prob of observation lying at ectream t_value
t_stat, p_value = stats.ttest_ind(setosa_sepal_length, versicolor_sepal_length)
print("\nT-test result for Sepal Length between Setosa and Versicolor:")
print(f"T-statistic: {t_stat}, p-value: {p_value}")
