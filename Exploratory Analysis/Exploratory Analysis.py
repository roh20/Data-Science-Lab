
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)


# In[113]:


#Importing Dataset
dataset = pd.read_csv('Automobile_data.csv')
print('Dataset first 5 rows head(5):\n',dataset.head(5))
print('Dataset last 5 rows tail(5):\n',dataset.tail(5))
print('Dataset column types:\n',dataset.dtypes)


# In[114]:


#Removing uncessary columns from dataset
dataset.drop(['num-of-doors','symboling','curb-weight','compression-ratio'],axis='columns',inplace=True)
dataset.head(2)


# In[115]:


#Renaming columns
dataset = dataset.rename(columns={"city-mpg":"city mileage(mpg)","highway-mpg":"highway mileage(mpg)"})
dataset.head(2)


# In[80]:


#Dropping duplicate rows
dataset.shape
duplicate_rows_df = dataset[dataset.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
print("Duplicate row:\n",duplicate_rows_df)
print("rows count",dataset.count())
dataset = dataset.drop_duplicates()
dataset.head(5)


# In[116]:


print('Checking dataset containing null values:\n',dataset.isnull().sum())


# In[117]:


#Fill drop,stroke,horsepower nan values with mean
dataset['bore'] = dataset['bore'].fillna(dataset['bore'].mean())
dataset['stroke'] = dataset['stroke'].fillna(dataset['stroke'].mean())
dataset['horsepower'] = dataset['horsepower'].fillna(dataset['horsepower'].mean())
dataset['normalized-losses'] = dataset['normalized-losses'].fillna(dataset['normalized-losses'].mean())
dataset['peak-rpm'] = dataset['peak-rpm'].fillna(dataset['peak-rpm'].mean())
dataset['price'] = dataset['price'].fillna(dataset['price'].mean())
print('Checking nan values count again:\n',dataset.isnull().sum())
#no nan values


# In[118]:


#detecting outliers
sns.boxplot(x=dataset['price'])


# In[119]:


#detecting outliers
sns.boxplot(x=dataset['peak-rpm'])


# In[120]:


#detecting outliers
sns.boxplot(x=dataset['city mileage(mpg)'])


# In[121]:


#detecting outliers
sns.boxplot(x=dataset['highway mileage(mpg)'])


# In[122]:


#Intrer quantile range
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
iqr = Q3-Q1
print(iqr)


# In[123]:


# Plotting a Histogram
dataset.make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make');


# In[106]:


# Finding the relations between the variables.
plt.figure(figsize=(20,10))
c= dataset.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c


# In[124]:


# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(dataset['horsepower'], dataset['price'])
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price')
plt.show()

