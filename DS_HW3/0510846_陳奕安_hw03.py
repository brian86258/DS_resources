#!/usr/bin/env python
# coding: utf-8

# # In this homework, I will preprocess some illegal data in my original dataset
# # Due the original dataset dosen't contain NaN, Null. So I randomly add some of these problematic conditions in original dataset
# 
# ## Start preprocess

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[34]:


raw_data=pd.read_csv('Automobile_data.csv')
print("The shape of this original dataset is {}".format(raw_data.shape))
# print(raw_data.head())
# print(raw_data.info())

print(raw_data.isnull().sum())


# # Though the original dataset doesn't contain Null ,NaN. However it does contain quite a few '?' data, which means nothing useful as well.
# 
# ## Following I will replace the '?' data with Nan. So that I can easily locate them.

# In[35]:


raw_data=raw_data.replace('?',np.NaN)
print(raw_data.isnull().sum())


# # And from the above analysis, we can find the features that needed to be modifeid. 
# ## {'normalized-losses', 'num-of-doors', 'curb-weight', 'num-of-clinders', 'bore', 'stroke' , 'horsepower' , 'peak-rpm' , 'price'}
# 
# ---
# Firstly, we deal with the normalized-losses

# In[40]:


wrong=raw_data[raw_data['normalized-losses']=='?']
empty=raw_data[raw_data['normalized-losses'].isnull()]
print(wrong.index)
print(empty.index)


#  ### We can firstly where are these null data. And then I choose to replace the empty data with the mean value of other non-empty data.

# In[41]:


a=raw_data[raw_data['normalized-losses'].notnull()]
b=(a['normalized-losses'].astype(int)).mean()
print(round(b))
raw_data['normalized-losses'].fillna(round(b),inplace=True)

empty=raw_data[raw_data['normalized-losses'].isnull()]
print(empty.index)


# ### And we can double check and find that there are no empty data in the ['normalized-losses']
# ---
# ### Then we move on to the 'num-of-doors' 

# In[42]:


print(raw_data['num-of-doors'])
print(raw_data[raw_data['num-of-doors'].isnull()].index)


# In[43]:


raw_data['num-of-doors'].fillna('four',inplace=True)
raw_data['num-of-doors']=raw_data['num-of-doors'].map({'two':2,'four':4})

print(raw_data['num-of-doors'])
print(raw_data[raw_data['num-of-doors'].isnull()].index)


# ### Because most of the cars have four doors, so I decide to replace the null data with 'four'. 
# ### And then because it will be easier to use numeric value, so I map all the strings to numeric values.
# 
# ---
# ### Then move on to 'curb-weight'.

# In[45]:


print('Max : ',raw_data['curb-weight'].max())
print('Min : ',raw_data['curb-weight'].min())
print('Medain :',raw_data['curb-weight'].median())


print(raw_data[raw_data['curb-weight'].isnull()].index)
raw_data.fillna(raw_data['curb-weight'].median(),inplace=True)
print(raw_data[raw_data['curb-weight'].isnull()].index)


# ### In this part, because the differenc between the maximum and minimum is rahter large. So I choose to use the medain to replace the empty value instead of mean value.
# 
# ---
# ### num-of-cylinder 

# In[39]:


print(raw_data['num-of-cylinders'].value_counts())


print(raw_data[raw_data['num-of-cylinders'].isnull()].index)
raw_data['num-of-cylinders'].fillna('four',inplace=True)
print(raw_data[raw_data['num-of-cylinders'].isnull()].index)


# ### We can find that the major number of cylinders is still 4, so I replace the empty data with four.
# ---
# ### Bore and Stroke : 
# ### As for Bore and Stroke I choose to use mean value to replace the empty value.

# In[38]:


clean_by_mean=['bore','stroke']

for name in clean_by_mean:
    print(name, raw_data[raw_data[name].isnull()].index)
    raw_data[name].fillna((raw_data[name].astype(float)).mean(),inplace=True)
    print(name, raw_data[raw_data[name].isnull()].index)


# ### Horsepower and peak-rpm :
# ### As for Horsepower and peak-rpm I choose to use mean value to replace the empty value.

# In[37]:


clean_by_median=['horsepower','peak-rpm']


for name in clean_by_median:
    print(name, raw_data[raw_data[name].isnull()].index)
    raw_data[name].fillna((raw_data[name]).median(),inplace=True)
    print(name, raw_data[raw_data[name].isnull()].index)


# ---
# ### Price :
# ### As for price, we can observe the range is also very wide, hence the median will be more suitable to represent. So I replace the empty data with the median value.

# In[36]:


a=raw_data[raw_data['price'].notnull()]
b=a['price'].astype(int)
print('Max : ',b.max())
print('Min : ',b.min())
print('Mean : :',b.mean())
print('Median : ',b.median())

print(raw_data[raw_data['price'].isnull()].index)
raw_data['price'].fillna(b.median(),inplace=True)
print(raw_data[raw_data['price'].isnull()].index)


# ## Final check, there is no empty or null data in this dataset.

# In[46]:


print(raw_data.isnull().sum())


# In[ ]:




