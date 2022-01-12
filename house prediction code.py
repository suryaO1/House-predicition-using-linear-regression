#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('train.csv')


# In[3]:


df.columns


# In[4]:


df =df.rename(columns={'BHK_NO.':'BHK_SIZE','TARGET(PRICE_IN_LACS)':'target'})


# In[5]:


df.head()


# In[6]:


pd.get_dummies(df.POSTED_BY)


# In[7]:


df.POSTED_BY.value_counts()


# In[8]:


df.head()


# In[9]:


pd.get_dummies(df.POSTED_BY)
hk = pd.get_dummies(df)
df['POSTED_BY'] = hk


# In[10]:


df.head()


# In[11]:


import math
df.SQUARE_FT.median()


# In[12]:


median_pa=math.floor(df.SQUARE_FT.median())
median_pa
df.SQUARE_FT.fillna(median_pa)


# In[20]:


numerical =[col for col in df.columns if df[col].dtype !='O']
print('there are {} numerical variables'.format(len(numerical)))


# In[21]:


print(numerical)


# In[22]:


pd.options.display.float_format = '{:,.4f}'.format
corr_matrix = df.corr()
corr_matrix


# In[23]:


corr_matrix['POSTED_BY'].sort_values(ascending=False)


# In[24]:


plt.figure(figsize=(16,10))
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt ='.2f', linecolor ='black')
plt.show()


# In[25]:


x = df[['BHK_SIZE','SQUARE_FT','RESALE','READY_TO_MOVE','LATITUDE','LONGITUDE','RERA']]
y = df[['target','POSTED_BY','UNDER_CONSTRUCTION',]]


# In[26]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=0)


# In[27]:


from sklearn.linear_model import LinearRegression
dlf = LinearRegression()


# In[28]:


dlf.fit(x_train,y_train)


# In[29]:


dlf.coef_


# In[30]:


dlf.predict(x_test)


# In[31]:


dlf.score(x_test,y_test)


# In[ ]:




