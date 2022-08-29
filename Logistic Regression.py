#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

#import and load iris dataset
from sklearn.datasets import load_iris
iris=load_iris()

#show the built_in description data
print(iris.DESCR)

#separate the dependent and independent variable
X=iris.data
Y=iris.target


#Exploratory Data Analysis
#features
iris_data = pd.DataFrame(X, columns =["Sepal Length","Sepal Width","Petal Length","Petal Width"])

#species/classes
iris_target=pd.DataFrame(Y, columns=["Species"])

#Replace values in our dependent variable with their species names
def Species_names(number):
    if number==0:
        return "Setosa"
    elif number==1:
       return "Versicolour"
    else:
       return "Virginica"
   
#Apply function to iris_target variable
iris_target["Species"]=iris_target["Species"].apply(Species_names)

#joining independent and dependent variables(with column )
full_iris=pd.concat([iris_data, iris_target],axis=1)

#Visualize
sns.pairplot(full_iris, hue="Species")

#Splitting our data to training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3,random_state=None)

#Logistic regression Classified/Algorithm
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

#fit the training model/fitting algorithm to the data
logreg.fit(X_train, Y_train)

#Predicting classes/species using X_test
Y_pred=logreg.predict(X_test)

#checking accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#Naive Bayes 
#Gaussian Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

#fit the training model
model.fit(X_train, Y_train)

#Predicting Species

Y_pred2=model.predict(X_test)

#Checking accurancy
print(accuracy_score(Y_test, Y_pred2))