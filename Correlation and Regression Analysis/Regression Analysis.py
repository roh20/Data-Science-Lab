
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[35]:


X = np.arange(5,65,10).reshape(-1,1)
y = np.array([5,18,15,34,43,33])


# In[36]:


model = LinearRegression()
model.fit(X,y)


# In[37]:


rSquare = model.score(X,y)
print('Coefficient of determination:',rSquare)


# In[39]:


print('\nIntercept:',model.intercept_)
print('\nSlope:',model.coef_)


# In[41]:


newModel = LinearRegression().fit(X,y.reshape(-1,1))
print('\nNew Intercept:',model.intercept_)
print('\nNew Slope:',model.coef_)


# In[43]:


#Predictions
y_pred = model.predict(X)
print('\nPrediction:\n',y_pred)


# In[45]:


y_pred = model.intercept_ + model.coef_ * X
print('\nPrediction using intercept and coefficient:\n',y_pred)


# In[49]:


XNew = np.arange(4).reshape(-1,1)
print('\nPrediction XNew:\n',model.predict(XNew))

