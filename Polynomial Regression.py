#Polynomial Regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load/read dataset
dataset= pd.read_csv("position_salaries.csv")

#separating dependent variables
X=dataset.iloc[:,1:2].values   #independent variable is level
Y=dataset.iloc[:,2].values   #dependent variable is salary

#fitting linear regression to dataset 
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X, Y)

#fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2) #transforming tool
poly.fit_transform(X)
X_poly=poly.fit_transform(X)

poly_reg= LinearRegression()
poly_reg.fit(X_poly, Y) #Polynomial model ready

#visualizing the linear regression results
plt.scatter(X, Y)
plt.plot(X, lin_reg.predict(X), color="red")
plt.title("Linear Regression")
plt.xlabel("Position level")
plt.ylabel("Predicted Salary")
plt.show()


#visualizing the polynomial regression results
plt.scatter(X, Y)
plt.plot(X, poly_reg.predict(poly.fit_transform(X)), color="green")
plt.title("Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("Predicted Salary")
plt.show()

#Optimizing our polynomial regression model by tuning our parameters
poly=PolynomialFeatures(degree=4)
X_poly=poly.fit_transform(X)
poly_reg= LinearRegression()
poly_reg.fit(X_poly, Y)

#Predicting new result with Linear Regression
 lin_reg.predict([[3.5]])
 lin_reg.predict([[11]])
 #Predicting new result with Linear Regression
 poly_reg.predict(poly.fit_transform([[3.5]]))
 poly_reg.predict(poly.fit_transform([[11]]))