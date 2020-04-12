
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


# In[12]:


player = 'Roger Federer'
df = pd.read_csv('player.csv',
                 parse_dates=['start date'],
                 dayfirst=True)


# In[13]:


print("Number of columns:", len(df.columns))
print("\nTail(5):")
print(df[df.columns[:4]].tail())


# In[14]:


npoints = df['player1 total points total']
points = df['player1 total points won'] / npoints
aces = df['player1 aces'] / npoints
fig, ax = plt.subplots(1, 1)
ax.plot(points, aces, '.')
ax.set_xlabel('% of points won')
ax.set_ylabel('% of aces')
ax.set_xlim(0., 1.)
ax.set_ylim(0.)


# In[15]:


df_bis = pd.DataFrame({'points': points,
                       'aces': aces}).dropna()
print('After dropping NaN values:\n')
print(df_bis.tail())


# In[17]:


print('\nCorrelation:\n',df_bis.corr())


# In[26]:


df_bis['result'] = (df_bis['points'] >
                    df_bis['points'].median())
df_bis['manyaces'] = (df_bis['aces'] >
                      df_bis['aces'].median())


# In[27]:

print('\nCross table:')
print(pd.crosstab(df_bis['result'], df_bis['manyaces']))


# In[28]:

print('\nChi square contingency:')
print(st.chi2_contingency(_))

