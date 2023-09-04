#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r"C:\Users\Dell\Downloads\telecom.csv")


# In[5]:


df['StreamingMovies'].unique()


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})


# In[14]:


df['gender']=df['gender'].map({'Male':0,'Female':1})


# In[15]:


df['StreamingTV']=df['StreamingTV'].map({'No':0,'Yes':1,'No internet service':2})


# In[16]:


df['StreamingMovies']=df['StreamingMovies'].map({'No':0,'Yes':1,'No internet service':2})


# In[17]:


df.head()


# In[18]:


df.corr()


# In[80]:


fig,ax=plt.subplots(1,3,figsize=(20,10))
ax[0].scatter(df['SeniorCitizen'],df['TotalCharges'])
ax[0].set_title('Seniorcitizen')
ax[1].scatter(df['StreamingTV'],df['TotalCharges'])
ax[1].set_title('streamingTv')
ax[2].scatter(df['StreamingMovies'],df['TotalCharges'])
ax[2].set_title('streamingmovies')


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[82]:


model=LinearRegression()


# In[87]:


X_train,X_test,Y_train,Y_test=train_test_split(df[['tenure']],df[['TotalCharges']],test_size=0.3,random_state=12)


# In[88]:


model.fit(X_train,Y_train)


# In[89]:


pred=model.predict(X_test)



# In[90]:


r2_score(Y_test,pred)


# In[93]:


np.sqrt(mean_squared_error(Y_test, pred))


# In[139]:


X_train1,X_test1,Y_train1,Y_test1=train_test_split(df[['MonthlyCharges','tenure']],df[['TotalCharges']],test_size=0.3,random_state=12)


# In[140]:


model.fit(X_train1,Y_train1)


# In[141]:


pred1=model.predict(X_test1)


# In[142]:


r2_score(Y_test1,pred1)


# In[143]:


np.sqrt(mean_squared_error(Y_test1, pred1))


# In[22]:


model=LogisticRegression()


# In[27]:


X_train2,X_test2,Y_train2,Y_test2=train_test_split(df[['gender']],df['Churn'],test_size=0.3,random_state=9)


# In[28]:


model.fit(X_train2,Y_train2)


# In[32]:


pred2=model.predict(X_test2)


# In[30]:


model.score(X_test2,Y_test2)


# In[31]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


# In[33]:


conf=confusion_matrix(Y_test2,pred2)


# In[35]:


conf


# In[34]:


disp=ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot()
plt.show()


# In[ ]:




