
# coding: utf-8

# In[10]:


#List 
print('List:')
languages = ['C', 'C++', 'Python']
languages = languages[0]
print(languages.title())


# In[11]:


#Tuple
print('Tuple:')
subjects = ('data science', 'data structures', 'data mining')
print("The subjects is: " + subjects[0])

print("\nThe available subjects are:")
for subject in subjects:
    print("- " + subject)


# In[12]:


#Set
names = ['abc','xyz','abc','xyz']
setName = set(names)
print('Set:',setName)


# In[14]:


import statistics as st
import numpy as np


# In[20]:



#Mean
arr = np.arange(4,100,4)
print(arr)
print('Mean:',st.mean(arr))


# In[26]:


#Mode
print('Mode:',st.mode([10,20,30,20,50,60]))


# In[28]:


#Median
print('Median:',st.median(arr))


# In[29]:


#Standard Deviation
print('Standard Deviation',st.stdev(arr))


# In[30]:


#Variance
print('Variance:',st.variance(arr))


# In[33]:


from scipy.stats import kurtosis 
import numpy as np  
import pylab as p  
  
x1 = np.linspace( -3, 3, 1000 ) 
y1 = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 
  
p.plot(x1, y1, '*') 
  
  
print('\nKurtosis for normal distribution :',kurtosis(y1)) 
  
print('\nKurtosis for normal distribution :',kurtosis(y1, fisher = False)) 
  
print( '\nKurtosis for normal distribution :',kurtosis(y1, fisher = True)) 


# In[39]:


#skewness of data
from scipy.stats import skew


# In[44]:


print('Skew for arr:',skew(arr))


# In[49]:


print('Skew for [2, 8, 0, 4, 1, 9, 9, 0:',skew([2, 8, 0, 4, 1, 9, 9, 0]))


# In[50]:


import pandas as pd
data = {'Subjects':['C', 'C++', 'Data Structures', 'TOC'],'Marks':[80,74,75,60]}
df = pd.DataFrame(data)
print('Data frame:',df)

