#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Run this cell
#Importing necessary libraries 
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import json
ans=[0]*5


# In[2]:


#Import the dataset and define the feature as well as the target datasets / columns 
df = pd.read_csv('zoo.csv')
#We drop the animal names since this is not a good feature to split the data on. 
df.drop(labels="animal_name",inplace=True,axis=1)
df_features = df.iloc[:,:-1]
df.describe()


# In[3]:


#Write a function to find the entropy on a split "target_col"
def entropy(target_col):
    freq = np.zeros(9)
    Entropy = 0
    tot = len(target_col)
    for val in target_col:
        freq[val]+=1
    unq = np.unique(target_col)
    for f in freq:
        if f!=0 and len(unq)!=1:
            Entropy-=(f/tot)*np.log(f/tot)/np.log(len(unq))
    return Entropy


# In[4]:


#Find the entropy of all the features in the dataset
#Save all the feature names in an array "feature names"
feature_names=['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone', 
               'breathes','venomous','fins','legs','tail','domestic','catsize']

feature_entropy = {}
# idx = 0
for feature in feature_names:
    feature_entropy[feature] = entropy(df[feature])
#     idx+=1
feature_entropy


# In[5]:


#Find the entropy of the feature "toothed"

ans[0]=feature_entropy["toothed"]
ans[0]


# In[6]:


#Write a function to calculate Information Gain on a split attribute and a target column
def InfoGain(data,split_attribute_name,target_name="class"):       
    #Calculate the entropy of the total dataset  
    orig_entropy = feature_entropy[target_name]
    #Calculate the values and the corresponding counts for the split attribute   
    split0 = df[target_name][(df[split_attribute_name]==0)]
    split1 = df[target_name][(df[split_attribute_name]==1)]
    #Calculate the weighted entropy  
    len0 = len(split0)
    len1 = len(split1)
    Len = len0+len1
    E0 = entropy(split0)
    E1 = entropy(split1)
    
    weighted_entropy = (len0/Len)*E0 + (len1/Len)*E1
    #Calculate the information gain
    Info_gain = orig_entropy - weighted_entropy
    return Info_gain


# In[7]:


#Find the information gain having split attribute "hair" and the target feature name "milk"

ans[1]=InfoGain(df,"hair","milk")
ans[1]


# In[8]:


#Find the Info gain having "milk" as the split attribute and all the other features as target features one at a time
for target_feature in feature_names:
    if target_feature!="milk":
        print("InfoGain[ milk,",target_feature,"]=",InfoGain(df,"milk",target_feature))


# In[9]:


# #Import Decision Tree Classifier from sklearn 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
#Split the given data into 80 percent training data and 20 percent testing data
X_train,X_test,Y_train,Y_test=train_test_split(df_features,df["class_type"],test_size=0.2,random_state=3)


# In[10]:


# Fit the given data
Classifier = DecisionTreeClassifier()
Classifier = Classifier.fit(X_train,Y_train)


# In[11]:


#Make a prediction on the test data and return the percentage of accuracy
Y_pred=Classifier.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
ans[2]=metrics.accuracy_score(Y_test, Y_pred)
ans[2]


# In[15]:


#Run this cell to visualize the decision tree
# from six import StringIO  
# from IPython.display import Image  
# from sklearn.tree import export_graphviz
# import pydotplus

from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(Classifier, out_file=dot_data, feature_names=feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[16]:


#Use sklearn to make a classification report and a confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print("CLASSIFICATION REPORT\n",classification_report(Y_test,Y_pred,labels=np.unique(Y_test),zero_division=1))
Confusion_matrix=pd.DataFrame(confusion_matrix(Y_test,Y_pred))
Confusion_matrix.index=[np.unique(Y_test)]
Confusion_matrix.columns=[np.unique(Y_test)]
Confusion_matrix


# In[17]:


#Find the recall,f1-score for class type '3'
ans[3] = [0.0,0.0]


# In[18]:


#Calculate Mean Absolute Error,Mean Squared Error and Root Mean Squared Error
Mean_absolute_error=metrics.mean_absolute_error(Y_test,Y_pred)
MSE=metrics.mean_squared_error(Y_test,Y_pred)
RMSE=metrics.mean_squared_error(Y_test,Y_pred,squared=False)
RMSE


# In[19]:


#Find the mean absolute error and root mean square error, save then in a list [mae,rmse]
ans[4]=[Mean_absolute_error,RMSE]
ans


# In[20]:


##do not change this code
import json
ans = [str(item) for item in ans]

filename = "umangpandey07@gmail.com_Umang_Pandey_DecisionTrees"


# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LinearRegression


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[21]:


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




