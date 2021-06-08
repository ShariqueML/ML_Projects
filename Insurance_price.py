#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Tasks:
# 1- Find Predictive Variables,
# 2- PREDICT THE DATA SET ON BASIC OF MALE AND FEMALE

# In[2]:


df = pd.read_csv('insurance.csv')
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


corr = df.corr()


# In[6]:


corr


# In[8]:


sns.heatmap(corr, annot=True)


# In[9]:





# In[10]:


df['sex'] = pd.get_dummies(df['sex'], drop_first=True)


# In[11]:


df['smoker'] = pd.get_dummies(df['smoker'], drop_first=True)
df['region'] = pd.get_dummies(df['region'], drop_first=True)


# In[12]:


df.head()


# In[13]:


corr1 = df.corr()


# In[14]:


corr1


# In[16]:


sns.heatmap(corr1, annot=True)


# So, charges is strongly positive correlated with smoker and also correlated with other features.
# Thus, our predicted variable is 'charges'.

# Let's build a linear regression model for the charges prediction.

# In[18]:


X = df.drop('charges', axis=1)
y = df['charges']


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[26]:





# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)


# In[20]:


X_train.shape


# In[21]:


X_test.shape


# In[27]:


x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)


# In[28]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[29]:


model.fit(x_train,y_train)


# In[31]:


model.score(x_test,y_test)


# In[32]:


y_pred = model.predict(x_test)


# In[33]:


from sklearn.metrics import r2_score,mean_squared_error
r2 = r2_score(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("coefficient of determination :", r2)
print("mean squared error :", mse)
print("root mean squared error :", rmse)


# So, our model accuracy is 79.7%.

# In[34]:


from sklearn.ensemble import RandomForestRegressor
rfr_model = RandomForestRegressor()
rfr_model.fit(x_train,y_train)


# In[35]:


rfr_model.score(x_test,y_test)


# In[36]:


# So, our rfr_model's accuracy is 87.8%.


# In[55]:


# Ltt's do parameter tunning to improve our model's accuracy.
rfr_model1 = RandomForestRegressor(n_estimators=1200, min_samples_split=12,min_samples_leaf=12,n_jobs=6)
rfr_model1.fit(x_train,y_train)


# In[56]:


rfr_model1.score(x_test,y_test)


# So, the accuracy of the model is increased by 10% which is good :)

# In[57]:


rfr_model1.score(x_train,y_train)


# In[ ]:


# As we can see, if we were tyring to improve our model's accuracy then there is a chance of overfitting.

