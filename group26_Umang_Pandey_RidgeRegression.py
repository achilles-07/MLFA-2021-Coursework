#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Do not make any changes in this cell
# Simply execute it and move on

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
ans = [0]*8


# In[2]:


# The exercise uses Boston housing dataset which is an inbuilt dataset of scikit learn.
# Run the cell below to import and get the information about the data.

# Do not make any changes in this cell.
# Simply execute it and move on

from sklearn.datasets import load_boston
boston=load_boston()
boston


# In[3]:


# Creating a dataframe

# Do not make any changes in this cell
# Simply execute it and move on

boston_df=pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.DataFrame(boston['target'])
boston_df


# In[4]:


# Question 1: Find the mean of the "target" values in the dataframe (boston_df)  
#             Assign the answer to ans[0] 
#             eg. ans[0] = 24.976534890123 (if mean obtained = 24.976534890123)


# In[5]:


# Your Code: Enter your Code below
target_mean = boston_df['target'].mean()


# In[6]:


#1 mark
ans[0] = target_mean


# In[7]:


# Just to get a look into distribution of data into datasets
# Plot a histogram for boston_df
boston_df.hist(figsize = (50,50))
plt.show()


# **Splitting the data using train_test_split from sklearn library**

# In[8]:


# Import machine learning libraries  for train_test_split

from sklearn.model_selection import train_test_split

# Split the data into X and Y

X = boston_df.iloc[:, :13].values
Y = boston_df['target']

# Spliting our data further into train and test (train-90% and test-10%)
# Use (randon_state = 42) in train_test_split, so that your answer can be evaluated

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.9, random_state = 42)



# **LINEAR REGRESSION**

# In[ ]:


# Question 2: Find mean squared error on the test set and the linear regression intercept(b)  
#             Assign the answer to ans[0] in the form of a list 
#             eg. ans[1] = [78.456398468,34.276498234098] 
#                  here , mean squared error             = 78.456398468
#                         linear regression intercept(b) = 34.276498234098


# In[9]:


# Fit a linear regression model on the above training data and find MSE over the test set.
# Your Code: Enter your Code below
from sklearn.linear_model import LinearRegression
from sklearn import metrics
LR = LinearRegression().fit(X_train, Y_train)
b = LR.intercept_
y_pred = LR.predict(X_test)
mse = metrics.mean_squared_error(Y_test, y_pred)


# In[10]:


# 2 marks
ans[1] = [mse, b]


# **RIDGE REGRESSION**

# In[ ]:


# Question 3: For what value of lambda (alpha)(in the list[0.5,1,5,10,50,100]) will we have least value of the mean squared error of testing set 
#             Take lambda (alpha) values as specified i.e. [0.5,1,5,10,50,100]
#             Assign the answer to ans[2]  
#             eg. ans[1] = 5  (if  lambda(alpha)=5)


# In[19]:


# Your Code: Enter your Code below
from sklearn.linear_model import Ridge
alpha = [0.5, 1, 5, 10, 50, 100]
ridge_temp_mse = np.zeros(shape=len(alpha))
idx = 0
for Lambda in alpha:
    rr = Ridge(alpha= Lambda)
    rr.fit(X_train, Y_train)
    y_pred = rr.predict(X_test)    
    ridge_temp_mse[idx] = (np.square(Y_test - y_pred)).mean()
    idx = idx+1
lambda_final = alpha[np.argmin(ridge_temp_mse)]


# In[20]:


#1 mark
ans[2] = lambda_final


# In[ ]:


# Question 4: Find mean squared error on the test set and the Ridge regression intercept(b)
#             Use the lamba(alpha) value obtained from question-3 
#             Assign the answer to ans[3] in the form of a list 
#             eg. ans[3] = [45.456398468,143.276498234098] 
#                  here , mean squared error             = 45.456398468
#                         Ridge regression intercept(b) = 143.276498234098


# In[21]:


# Your Code: Enter your Code below
Ridge_reg = Ridge(ans[2])
Ridge_reg.fit(X_train, Y_train)
y_pred_ridge = Ridge_reg.predict(X_test)
mse_ridge = metrics.mean_squared_error(Y_test, y_pred_ridge)
b_ridge = Ridge_reg.intercept_


# In[22]:


# 2 marks
ans[3] = [mse_ridge, b_ridge]


# In[23]:


# Plot the coefficient of the features( CRIM , INDUS , NOX ) with respective to  the lambda values specified [0.5,1,5,10,50,100]
# Enter your code below
coefs = []
alphas = [0.5,1,5,10,50,100]
for lam in alphas:
    ridge = Ridge(alpha = lam, fit_intercept = False)
    ridge.fit(X_train[:,[0, 2, 4]], Y_train)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas, coefs)
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.legend(['CRIM', 'INDUS', 'NOX'])
plt.axis('tight')
plt.show()


# **LASSO REGRESSION**

# In[ ]:


# Question 5: For lambda (alpha)=1 find the lasso regression intercept and the test set mean squared error
#             Assign the answer to ans[4] in the form of a list 
#             eg. ans[4] = [35.456398468,14.276498234098] 
#                  here , mean squared error             = 35.456398468
#                         lasso regression intercept(b) = 14.276498234098


# In[24]:


# Your Code: Enter your Code below
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha = 1)
lasso_reg.fit(X_train, Y_train)
y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = metrics.mean_squared_error(Y_test, y_pred_lasso)
b_lasso = lasso_reg.intercept_


# In[25]:


#2 mark
ans[4] = [mse_lasso, b_lasso]


# In[26]:


# Question 6: Find the most  important feature  in the data set i.e. which feature coefficient is further most non zero if lambda is increased gradually
#             let CRIM=1,	ZN=2, INDUS=3,	CHAS=4,	NOX=5,	RM=6,	AGE=7,	DIS=8,	RAD=9,	TAX=10,	PTRATIO=11,	B=12,	LSTAT=13
#              eg. if your answer is "CHAS"
#                   then your answer should be ans[5]=4


# In[28]:


# Your Code: Enter your Code below
max_coeff = -1
lmbd_arr = np.linspace(0.1,100,num=1000)
for lmbd in lmbd_arr:
    lasso_reg_temp = Lasso(alpha = lmbd)
    lasso_reg_temp.fit(X_train, Y_train)
    max_coeff = np.argmax(abs(lasso_reg_temp.coef_))
# max_coeff


# In[30]:


#2 marks
ans[5] = max_coeff+1


# Run the below cell only once u complete answering all the above answers 
# 

# In[31]:


##do not change this code
import json
ans = [str(item) for item in ans]

filename = "umangpandey07@gmail.com_Umang_Pandey_RidgeRegression"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LinearRegression


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[32]:


from importlib import import_module
import os
from pprint import pprint

findScore = import_module('findScore')
response = findScore.main(ans)
response['details'] = filename
with open(f'evaluation_{filename}.json', 'w') as outfile:
    json.dump(response, outfile)
pprint(response)


# In[ ]:




