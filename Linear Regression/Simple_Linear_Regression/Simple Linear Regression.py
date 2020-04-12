
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
print('Dataset info')
print(dataset.info())
print('Checking null values')
print(dataset.isnull())
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[3]:


print("Dataset Shape:")
print(dataset.shape)
print("\n\nDataset head:\n")
print(dataset.head)


# In[4]:


# Splitting the dataset into the Train and Test set 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[5]:


# Fitting Model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


# Predicting the Test set results
y_pred = model.predict(X_test)


# In[7]:


# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[8]:


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, model.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[9]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print('Mean Absolute square error:',mean_absolute_error(y_test,y_pred))
print('Mean squared error:', mean_squared_error(y_test,y_pred))
print('Variance score:',r2_score(y_test,y_pred))
print('Training Accuracy:',model.coef_)
print('Intercept:',model.intercept_)

