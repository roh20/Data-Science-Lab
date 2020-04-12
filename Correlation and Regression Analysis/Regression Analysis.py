
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


print('Intercept:',model.intercept_)
print('Slope:',model.coef_)


# In[41]:


newModel = LinearRegression().fit(X,y.reshape(-1,1))
print('New Intercept:',model.intercept_)
print('New Slope:',model.coef_)


# In[43]:


#Predictions
y_pred = model.predict(X)
print('Predictions:\n',y_pred)


# In[45]:


y_pred = model.intercept_ + model.coef_ * X
print('Predictions:\n',y_pred)


# In[49]:


XNew = np.arange(4).reshape(-1,1)
print('Predictions:\n',model.predict(XNew))

