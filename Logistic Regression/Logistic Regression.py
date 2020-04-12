
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[96]:


#Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print('Dataset Head(5):')
print(dataset.head(5))


# In[97]:


print("Dataset informartion:\n")
dataset.info()
print("\n\nDataset null info:\n")
print(dataset.isnull().sum())


# In[98]:


# Storing independent and dependent variable
X = dataset.iloc[:,[2,3]]
y = dataset.iloc[:,-1]
# In[99]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[100]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[101]:


# Fitting Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)


# In[102]:


# Predicting the Test set results
y_pred = model.predict(X_test)
print('Predictions:\n',y_pred)


# In[103]:


# Confusion Matrix and classification report
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Classification Report :\n',classification_report(y_test,y_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))
import seaborn as sns
sns.heatmap(confusion_matrix(y_test,y_pred), annot = True, cbar = True,xticklabels= ['Age','Salary'],yticklabels =['Purchased','Not Purchased'])


# In[104]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[105]:


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('black', 'yellow'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[106]:


from sklearn.metrics import  mean_absolute_error,mean_squared_error, mean_squared_log_error, r2_score
import sklearn.metrics as mt
print('Mean Absoulute Score:',mean_absolute_error(y_test, y_pred)*100)
print('Mean Squared Error:',mean_squared_error(y_test,y_pred)*100)
print('Mean Squared Log Error:',mean_squared_log_error(y_test,y_pred)*100)
print('R2 Score:',r2_score(y_test,y_pred)*100)
print("Accuracy:",mt.accuracy_score(y_test, y_pred))

