#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Emre Kılavuz 220201019

'''

import pandas as pd
import numpy as np

# Reading csv file with a delimiter
d = pd.read_csv('student-mat.csv', delimiter=";")
# Choose the features to be used in the bayes algorithm according to my observation
training_features = d.iloc[:,[6,7,13,14,28]]

# Create a list of two classes as who passed the course as 1 and who fail is 0 in order to reduce the target classes
pass_the_course_list = []
final_grades = d.iloc[:,-1]
for item in final_grades:
    if(item >= 10):
        pass_the_course_list.append(1)
    else:
        pass_the_course_list.append(0)
# Convert the target list to numpy one dimensional array of integers
target_feature = pd.DataFrame(pass_the_course_list, columns=['target'])
Target_feature = target_feature.values
Target_feature2 = Target_feature.flatten()

# Get two categorical column from the initial data frame and get them as dummies
encoded_fjob = pd.get_dummies(d.Fjob)
encoded_reason = pd.get_dummies(d.reason)
# Change the column names because some of them is the same with each other
encoded_fjob = encoded_fjob.rename(columns = {'other' : 'otherFjob', 'health' : 'healthFjob'})
encoded_reason = encoded_reason.rename(columns = {'other' : 'otherReason'})
# Concatenate the training features and convert them to numpy array
concatenated_train_set = pd.concat([training_features,encoded_fjob,encoded_reason],axis='columns')
Concatenated_train_set = concatenated_train_set.values

# Split the training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Concatenated_train_set,Target_feature2,test_size = 0.20,random_state=123)

# Scale the training features to make easier to fit the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test) 

# Import the Bayes class which from the other file
from bayes_train_predict import Bayes
bayes_instance = Bayes()
#Fit and predict
bayes_instance.fit(X_train, y_train)
predictions = bayes_instance.predict(X_test)
print(predictions)
print("y test : ",y_test)

# Calculate the accuracy of predictions function
def prediction_accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) /len(y_true)
    return accuracy

accuracy = prediction_accuracy(y_test, predictions)
print("Accuracy is : ", accuracy) 

# Determine contingency table
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

# Calculating value of contingency table cells
for i in range(len(predictions)):
	if predictions[i] == 0 and (y_test[i] == 0):
		true_negative += 1
	elif predictions[i] == 0 and (y_test[i] == 1):
		false_negative += 1
	elif predictions[i] == 1 and (y_test[i] == 1):
		true_positive += 1
	elif predictions[i] == 1 and (y_test[i] == 0):
		false_positive += 1


# Cool printing
print(str("\n   ")+str(1)+  "  |"  +str("  ")+str(  0))
print(str(1)+ "| " +str(true_positive) +str("   ")+  str(false_negative))
print(str(0)+ "| " +str(false_positive) + str("   ")+  str(true_negative))

# Calculate the jaccard coefficient
jaccard_coefficient = true_positive/(false_positive + true_positive + false_negative)

print("\nJaccard Coefficient: ",jaccard_coefficient)







