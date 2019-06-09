#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('data/train.csv')


# ## Cleaning

# #### Processing Survived

# In[3]:


df.rename(index=str, columns={'Survived': 'survived'}, inplace=True)


# #### Change Sex column to 'is_male' boolean

# In[4]:


df.rename(index=str, columns={'Sex': 'is_male'}, inplace=True)
df.is_male = (df.is_male == 'male').map(int);


# #### Removing passengers that paid no fare

# In[5]:


df.rename(index=str, columns={'Fare': 'fare'}, inplace=True)
df = df.loc[df.fare != 0, :]


# #### Processing age feature
# 
# _"Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5"

# In[6]:


df.rename(index=str, columns={'Age': 'age'}, inplace=True)

df = df.loc[df.isnull().age.map(lambda b: not b),:]

def handle_estimated_age(n):
    return int(n)

df.age = df.age.map(handle_estimated_age)

df.age.unique()
#df.loc[df.Age.map(lambda x: x != int(x)),:]


# ## Graphing

# #### Selecting for columns relevant to graph

# In[7]:


df = df[['survived', 'is_male', 'age', 'fare']]


# In[8]:



df_survived = df.loc[df.survived == 1, :]
df_kicked_the_bucket = df.loc[df.survived != 1, :]


plt.figure(figsize=(15,10))
plt.scatter(df_survived.age, df_survived.fare, s=5, color='blue')
plt.scatter(df_kicked_the_bucket.age, df_kicked_the_bucket.fare, s=5, color='red')
plt.xlabel('Age (years)', fontsize=20)
plt.ylabel('Fare ($)', fontsize=20)
plt.title('Ticket Price vs. Age', fontsize=20)


# ## Data Check

# In[9]:


df.head(10)


# In[10]:


df.describe()


# In[ ]:




