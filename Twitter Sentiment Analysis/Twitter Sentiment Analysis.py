
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np


# In[55]:


dataset = pd.read_csv('twitter data.csv')
print('Dataset info:\n',dataset.info())
print('\nNull Values info:\n',dataset.isnull().sum())
dataset = dataset.dropna()
print('\nAfter droping NaN Values info:\n',dataset.isnull().sum())
X = dataset.SentimentText
y = dataset.Sentiment


# In[56]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[57]:


#Corpus
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=100,stop_words='english',analyzer='word',lowercase=True)#,token_pattern='[^a-zA-Z]')
X_train_counts = count_vect.fit_transform(X_train)


# In[58]:


#Using Tfid for more accuracy
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# In[59]:


#Fitting model
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf,y_train)


# In[60]:


#K-fold cross validation
import sklearn.model_selection as ms
seed=7
kfold = ms.KFold(n_splits=10, random_state=seed)
results = ms.cross_val_score(clf,X_train_tfidf , y_train, cv=kfold)


# In[61]:


print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[62]:


#Predicting Results
X_new_counts = count_vect.transform(X_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)#use the tfidif or countvectorizer to do the predictions
np.mean(predicted == y_test)


# In[63]:


predictions = []
for prediction in predicted:
    if prediction == 0:
        predictions.append('Negative')
    else:
        predictions.append('Positive')


# In[53]:


#Predictions Visualization
import matplotlib.pyplot as plotter 
pieLabels = 'Positive Tweets','Negative Tweets'
predictionPercentage = [predictions.count('Positive'),predictions.count('Negative')] 
figureObject, axesObject = plotter.subplots()

axesObject.pie(predictionPercentage,
        labels=pieLabels,
        autopct='%1.2f',
        startangle=90)
axesObject.axis('equal')
plotter.show()

