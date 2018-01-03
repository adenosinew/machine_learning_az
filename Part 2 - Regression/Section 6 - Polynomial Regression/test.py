#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:46:17 2018

@author: kexing
"""

# Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/media/kexing/DATA/CodePractice/machine_learning_az/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


plt.scatter(x,y,color='red')
plt.plot(x, lin_reg.predict(x),color='blue')
plt.title("Truth or bluff(linear regression)")
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()


plt.scatter(x,y,color='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title("Truth or bluff(poly regression)")
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()