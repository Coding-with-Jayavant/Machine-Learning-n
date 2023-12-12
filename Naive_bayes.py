# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:25:39 2023

@author: jwark
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn import metrics

wine = datasets.load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train,  X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=50)

nb_classifier = GaussianNB()

# Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = nb_classifier.predict(X_test)
y_pred
# Print the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate and print the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate and print the F1 score
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score: {f1:.2f}")

class_report = metrics.classification_report(y_test, y_pred)

print('\nClassification Report:')
print(class_report)
