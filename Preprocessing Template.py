# Preprocessing template

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing the dataset
dataset = pd.read_csv('org_data.csv')

# segregagting the independent values from the dependent values
# X, independent values and Y, dependent values
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 4].values


#Estimating missing values
from sklearn.impute import SimpleImputer
imputed_values = SimpleImputer(missing_values=np.nan, strategy="mean")
imputed_values = imputed_values.fit(X[:,1:3])
X[:,1:3] = imputed_values.transform(X[:, 1:3]) #missing values get transformed with the mean of their columns


# Encoding Categorical data
#Encoding independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
ct = make_column_transformer((OneHotEncoder(),[0]), remainder='passthrough') #remaining columns won't be affected hence "passthrough"
X = ct.fit_transform(X)


#Encoding dependent variable
le = LabelEncoder()
Y = le.fit_transform(Y)


# spliting our data to training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

