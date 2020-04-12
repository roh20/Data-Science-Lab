
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[88]:


#Importing dataset
dataset = pd.read_csv('iris.csv')
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
dataset['species'] = lb.fit_transform(dataset[ 'species'])


# In[89]:


print("Dataset informartion:")
dataset.info()
print("\n\nDataset null info:")
print(dataset.isnull().sum())


# In[90]:


# Storing independent and dependent variable
X = dataset.iloc[:,0:-1]
y = dataset.iloc[:,-1]
print('\nHead(5) values for X and y:')
print(X.head(5))
print(y.head(5))


# In[91]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[92]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[93]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


# In[94]:


# Predicting the Test set results
y_pred = model.predict(X_test)
print('\nPredictions:\n',y_pred)


# In[95]:


# Confusion Matrix and classification report
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('\nClassification Report :\n',classification_report(y_test,y_pred))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,y_pred))
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred), annot = True, cbar = True,xticklabels= ['sestosa','vsersi','virginica'],yticklabels =['sestosa','vsersi','virginica'])


# In[96]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import sklearn.metrics as mt
print('\nMean Absolute square error:',mean_absolute_error(y_test,y_pred))
print('Mean squared error:', mean_squared_error(y_test,y_pred))
print('Variance score:',r2_score(y_test,y_pred))
print("Accuracy:",mt.accuracy_score(y_test, y_pred))

