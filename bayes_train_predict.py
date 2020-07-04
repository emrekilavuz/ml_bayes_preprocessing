#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:34:05 2020

@author: Emre Kılavuz 220201019
"""

import numpy as np

class Bayes:
    
    def fit(self, training_features, y):
        #Define class variables
        number_of_samples, number_of_featues = training_features.shape
        self._classes = np.unique(y)
        number_of_classes = len(self._classes)
        
        self._mean = np.zeros((number_of_classes, number_of_featues), dtype=np.float64)
        self._variance = np.zeros((number_of_classes, number_of_featues), dtype=np.float64)
        self._priors = np.zeros(number_of_classes, dtype=np.float64)
        
        # Calculate mean, variance and prior probability
        for c in self._classes:
            #print("y esittir : ",y)
            #print("c esittir : ",c)
            X_c = training_features[y==c] 
            #print(X_c)      
            self._mean[c, :] = X_c.mean(axis=0)
            self._variance[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(number_of_samples)
            
            
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    # Calculate posterior probabilities and take the maximum value
    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._probability_density(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]
    
    # Calculate probability density function
    def _probability_density(self, class_idx, x):
        mean = self._mean[class_idx]
        variance = self._variance[class_idx]
        #print("x is :",x)
        #print("mean is : ",mean)
        numerator = np.exp(- (x-mean)**2 / (2*variance))
        denominator = np.sqrt(2* np.pi * variance)
        return numerator / denominator
    
    
    