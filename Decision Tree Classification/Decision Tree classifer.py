
# coding: utf-8

# In[90]:


import numpy as pd
import matplotlib.pyplot as plt
import pandas as pd


# In[91]:


#Reading data
dataset = pd.read_csv('titanic_data.csv')
dataset.head(5)


# In[92]:


#Removing unnecessary columns
dataset.drop(['PassengerId','Name','Ticket','fair','Embarked','Cabin'],axis = 'columns',inplace = True)
dataset.head(5)


# In[93]:


#Fill na values with mean
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset.head(5)
print("Dataset informartion:\n")
dataset.info()
print("\n\nDataset null info:\n")
dataset.isnull()


# In[94]:


# Storing independent and dependent variable
y = dataset.Survived
dataset.drop(['Survived'],axis = 'columns',inplace = True)
X = dataset


# In[95]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()

X['Sex'] = lb.fit_transform(X[ 'Sex'])
X.head()


# In[96]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X_train.head(5)


# In[97]:


# Fitting model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[98]:


#Prediction
y_pred = classifier.predict(X_test)
y_pred
print('Accuracy:',classifier.score(X_test,y_test)*100)


# In[99]:


#drawing Decision Tree, you need graphviz application installed(with environment variable set) to print decision tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[100]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Classification Report :\n',classification_report(y_test,y_pred))
print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))


# In[101]:


from sklearn.metrics import  mean_absolute_error,mean_squared_error, mean_squared_log_error, r2_score
import sklearn.metrics as mt
print('Mean Absoulute Score:',mean_absolute_error(y_test, y_pred)*100)
print('Mean Squared Error:',mean_squared_error(y_test,y_pred)*100)
print('Mean Squared Log Error:',mean_squared_log_error(y_test,y_pred)*100)
print('R2 Score:',r2_score(y_test,y_pred)*100)
print("Accuracy:",mt.accuracy_score(y_test, y_pred))

