#!/usr/bin/env python
# coding: utf-8

# # GRIP : THE SPARKS FOUNDATION
# Data Science and Business Analytics Intern
# 
# Author : Riya Gaba
# 
# Task 1 : Prediction using Supervised ML

# # importing libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# # Reading data from url

# In[2]:


url="http://bit.ly/w-data"
data = pd.read_csv(url)


# In[3]:


print(data.shape)
data.head()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.plot(kind="scatter",x="Hours",y="Scores",)


# In[7]:


data.corr(method="pearson")


# In[8]:


data.corr(method="spearman")


# In[9]:


hours=data['Hours']
scores=data['Scores']


# In[10]:


sns.displot(hours)


# In[11]:


sns.displot(scores)


# ### Linear Regression

# In[12]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=50)


# In[14]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression() 
reg.fit(X_train,Y_train)


# In[15]:


m = reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[16]:


y_pred=reg.predict(X_test)


# In[17]:


actual_predicted=pd.DataFrame({'Target': Y_test,'Predicted' : y_pred})
actual_predicted


# In[18]:


sns.set_style('whitegrid')
sns.displot(np.array(Y_test-y_pred))
plt.show()


# ### What would be the predicted score if a student student studies for 9.25 hours/day?

# In[19]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} in exam." .format(h,s))


# ### MODEL EVALUATION

# In[20]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test,y_pred))
print('R2 Score:',r2_score(Y_test,y_pred))


# In[ ]:




