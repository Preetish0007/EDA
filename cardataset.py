#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


df=pd.read_csv(r"C:\Users\Dell\Downloads\CarDataSet.csv")


# In[26]:


df.drop('Unnamed: 0',axis = 1,inplace = True)


# In[27]:


kmkg = 0
kmpl = 0
for i in df.Mileage:
    if str(i).endswith("km/kg"):
        kmkg+=1
    elif str(i).endswith("kmpl"):
        kmpl+=1
print('The number of rows with Km/Kg : {} '.format(kmkg))
print('The number of rows with Kmpl : {} '.format(kmpl))


# In[28]:


df.drop('New_Price',axis=1,inplace=True)
df.dropna(inplace=True,axis=0)


# In[29]:


df


# In[30]:


df= df[df['Mileage']!= 'nan']


# In[31]:


df.isnull().sum()


# In[32]:


Correct_Mileage= []
for i in df.Mileage:
    if str(i).endswith('km/kg'):
        i = i[:-6]
        i = float(i)*1.40
        Correct_Mileage.append(float(i))
    elif str(i).endswith('kmpl'):
        i = i[:-6]
        #print(i)
        Correct_Mileage.append(float(i))


# In[33]:


df['Mileage']=Correct_Mileage


# In[34]:


df.head()


# In[35]:


plt.scatter(df['Year'],df['Kilometers_Driven'])
plt.show()

# removing outlier


# In[36]:


df = df[df['Kilometers_Driven']<6000000]
#outlier removed


# In[37]:


plt.scatter(df['Year'],df['Kilometers_Driven'])
plt.show()


# In[38]:


df.info()


# In[39]:


df.head()


# In[44]:


fig,ax=plt.subplots(1,3,figsize=(20,5))
ax[0].scatter(df['Mileage'],df['Price'])
ax[1].scatter(df['Fuel_Type'],df['Price'])
ax[2].scatter(df['Transmission'],df['Price'])


# In[45]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[46]:


model=LinearRegression()


# In[86]:


X_train,X_test,Y_train,Y_test=train_test_split(df[['Mileage']],df[['Price']],test_size=0.3,random_state=12)


# In[87]:


model.fit(X_train,Y_train)


# In[88]:


pred=model.predict(X_test)


# In[89]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[90]:


r2_score(Y_test,pred)


# In[91]:


np.sqrt(mean_squared_error(Y_test,pred))


# In[61]:


m=model.coef_


# In[62]:


c=model.intercept_


# In[63]:


m


# In[64]:


c


# In[65]:


#y=-0.67x + 21.56


# In[60]:


df.corr()


# In[85]:


X_train1,X_test1,Y_train1,Y_test1=train_test_split(df[['Mileage']],df[['Price']],test_size=0.3,random_state=12)


# In[66]:


pred


# In[95]:


result2=pd.DataFrame()
result2['Mileage']=X_test['Mileage']
result2['Actual']=Y_test['Price']
result2['predicted']=pred
result2.head()


# In[99]:


result2['error']=abs(result2['Actual']-result2['predicted'])


# In[106]:


result2['error %']=result2['error']/result2['Actual']*100


# In[107]:


result2


# In[ ]:





# In[ ]:




