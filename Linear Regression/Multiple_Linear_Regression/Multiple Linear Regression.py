
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[8]:


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
print('Dataset info:\n')
print(dataset.info())
print('\n\nChecking null values:\n')
print(dataset.isnull().sum())
X = dataset.iloc[:, :-1].values # all columns except last
y = dataset.iloc[:, 4].values   #last column


# In[9]:


print("Dataset Shape:")
print(dataset.shape)
print("\n\nDataset head:\n")
print(dataset.head(5))


# In[10]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# In[11]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[12]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[13]:


# Predicting the Test set results
y_pred = model.predict(X_test)


# In[14]:

print('\nPredictions:\n',y_pred)

# In[15]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print('\nMean Absolute square error:',mean_absolute_error(y_test,y_pred))
print('Mean squared error:', mean_squared_error(y_test,y_pred))
print('Variance score:',r2_score(y_test,y_pred))
print('Training Accuracy:',model.coef_)
print('Intercept:',model.intercept_)

