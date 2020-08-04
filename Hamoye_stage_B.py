#!/usr/bin/env python
# coding: utf-8

# ## Dataset Description
# 
# The dataset for the remainder of this quiz is the Appliances Energy Prediction data. The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters). The attribute information can be seen below.
# 
# Attribute Information:
# 
# Date, time year-month-day hour:minute:second
# 
# Appliances, energy use in Wh
# 
# lights, energy use of light fixtures in the house in Wh
# 
# T1, Temperature in kitchen area, in Celsius
# 
# RH_1, Humidity in kitchen area, in %
# 
# T2, Temperature in living room area, in Celsius
# 
# RH_2, Humidity in living room area, in %
# 
# T3, Temperature in laundry room area
# 
# RH_3, Humidity in laundry room area, in %
# 
# T4, Temperature in office room, in Celsius
# 
# RH_4, Humidity in office room, in %
# 
# T5, Temperature in bathroom, in Celsius
# 
# RH_5, Humidity in bathroom, in %
# 
# T6, Temperature outside the building (north side), in Celsius
# 
# RH_6, Humidity outside the building (north side), in %
# 
# T7, Temperature in ironing room , in Celsius
# 
# RH_7, Humidity in ironing room, in %
# 
# T8, Temperature in teenager room 2, in Celsius
# 
# RH_8, Humidity in teenager room 2, in %
# 
# T9, Temperature in parents room, in Celsius
# 
# RH_9, Humidity in parents room, in %
# 
# To, Temperature outside (from Chievres weather station), in Celsius
# 
# Pressure (from Chievres weather station), in mm Hg
# 
# RH_out, Humidity outside (from Chievres weather station), in %
# 
# Wind speed (from Chievres weather station), in m/s
# 
# Visibility (from Chievres weather station), in km
# 
# Tdewpoint (from Chievres weather station), Â °C
# 
# rv1, Random variable 1, nondimensional
# 
# rv2, Random variable 2, nondimensional
# 
# To answer some questions, you will need to normalize the dataset using the MinMaxScaler after removing the following columns: [“date”, “lights”]. The target variable is “Appliances”. Use a 70-30 train-test set split with a  random state of 42 (for reproducibility). Run a multiple linear regression using the training set and evaluate your model on the test set.

# In[46]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


df=pd.read_csv('downloads\energydata_complete.csv')


# In[48]:


df.head()


# In[49]:


df.drop(columns=['date', 'lights'], inplace=True) 


# In[50]:


df.head()


# In[51]:


df.isnull().sum()


# In[52]:


#Firstly, we normalise our dataset to a common scale using the min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(columns=['Appliances'])
heating_target = normalised_df['Appliances']


# In[53]:


predictor = df["T2"].values.reshape(-1,1)
response = df["T6"].values.reshape(-1,1)


# In[54]:


#Now, we split our dataset into the training and testing dataset. Recall that we had earlier segmented the features and target variables.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictor, response, test_size=0.3, random_state=42)


# In[55]:


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

#fit the model to the training dataset
linear_model.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model.predict(x_test)


# In[56]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# ## MLR

# In[62]:


#Now, we split our dataset into the training and testing dataset. Recall that we had earlier segmented the features and target variables.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_df, heating_target, test_size=0.3, random_state=42)


# In[63]:


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

#fit the model to the training dataset
linear_model.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model.predict(x_test)


# In[64]:


print(linear_model.intercept_)


# ## Measuring Regression performance

# In[65]:


from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 3)


# In[67]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# In[68]:


#RSS
rss = np.sum(np.square(y_test - predicted_values))
round(rss, 2)


# In[69]:


#RMS
from sklearn.metrics import  mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 2) 


# In[70]:


# coefficient of determination = r-squared

from sklearn.metrics import r2_score
r2_score = r2_score(y_test, predicted_values)
round(r2_score, 2)


# ## Ridge Regression

# In[79]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# ## Lasso Regression

# In[86]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[81]:


#comparing the effects of regularisation
def get_weights_df(model, feat, df):
    #this function returns the weight of every feature

    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', df]
    weights_df[df].round(3)
    return weights_df


# In[82]:


linear_model_weights = get_weights_df(linear_model, x_train, 'Linear_Model_Weight')
ridge_weights_df = get_weights_df(ridge_reg, x_train, 'Ridge_Weight')
lasso_weights_df = get_weights_df(lasso_reg, x_train, 'Lasso_weight')


# In[83]:


final_weights = pd.merge(linear_model_weights, ridge_weights_df, on='Features')
final_weights = pd.merge(final_weights, lasso_weights_df, on='Features')


# In[84]:


final_weights


# In[87]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[88]:


#comparing the effects of regularisation
def get_weights_df(model, feat, df):
    #this function returns the weight of every feature

    weights = pd.Series(model.coef_, feat.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', df]
    weights_df[df].round(3)
    return weights_df


# In[90]:


#RMS
from sklearn.metrics import  mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predicted_values))
round(rmse, 3)


# In[ ]:




