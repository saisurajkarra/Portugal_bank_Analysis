#!/usr/bin/env python
# coding: utf-8

# In[28]:


# importing all the libraries    
import pandas as pd
import numpy as np
import seaborn as sn
    
#read the csv file and store it in 'bank' dataframe
bank = pd.read_csv('bank_data.csv')
bank.head()


# In[30]:


# list all columns (for reference)
bank.columns


# In[31]:


# convert the response to numeric values and store as a new column
bank['outcome'] = bank.default.map({'no':0, 'yes':1})


# In[32]:


get_ipython().run_line_magic('matplotlib', 'inline')
# creating boxplot pr age and outcome
bank.boxplot(column='age', by='outcome')


# In[33]:



bank.groupby('job').outcome.mean()


# In[34]:


# create job_dummies 
job_dummies = pd.get_dummies(bank.job, prefix='job')
job_dummies.drop(job_dummies.columns[0], axis=1, inplace=True)


# In[35]:


bank.groupby('default').outcome.mean()


# In[36]:


# but only one person in the dataset has a status of yes
bank.default.value_counts()


# In[37]:


# so, let's treat this as a 2-class feature rather than a 3-class feature
bank['default'] = bank.default.map({'no':0, 'unknown':1, 'yes':1})


# In[38]:


# convert the feature to numeric values
bank['contact'] = bank.contact.map({'cellular':0, 'telephone':1})


# In[39]:


# looks like a useful feature at first glance
bank.groupby('month').outcome.mean()


# In[40]:


bank.groupby('month').outcome.agg(['count', 'mean']).sort_values('count')


# In[41]:


# boxployt for duration and outcome
bank.boxplot(column='duration', by='outcome')


# In[42]:


# looks like a useful feature
bank.groupby('previous').outcome.mean()


# In[43]:


# looks like a useful feature
bank.groupby('poutcome').outcome.mean()


# In[44]:


# create poutcome_dummies
poutcome_dummies = pd.get_dummies(bank.poutcome, prefix='poutcome')
poutcome_dummies.drop(poutcome_dummies.columns[0], axis=1, inplace=True)
# concatenate bank DataFrame with job_dummies and poutcome_dummies
bank = pd.concat([bank, job_dummies, poutcome_dummies], axis=1)


# In[65]:


## create X dataframe having 'default', 'Target', 'previous' and including 13 dummy #columns
feature_cols = ['default','previous'] + list(bank.columns[-13:])
X = bank[feature_cols]
# create y
y = bank.outcome
X.head()


# In[66]:


#convert the response to numeric values and store as a new column
bank['Target'] = bank.Target.map({'no':0, 'yes':1})


# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=12)
# calculate cross-validated AUC
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=5)
model.fit(X_train,y_train)


# In[68]:


#Store the predicted data in 'predicted' array
predicted = model.predict(X_test)
# Import metrics
from sklearn import metrics
# generate evaluation metrics-
print(metrics.accuracy_score(y_test, predicted))


# In[69]:


# Print out the confusion matrix
print(metrics.confusion_matrix(y_test, predicted))


# In[70]:


# Print out the classification report, and check the f1 score
print(metrics.classification_report(y_test, predicted))


# In[82]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
predicted= model.predict(X_test) 


# In[83]:


print(metrics.classification_report(y_test, predicted))


# In[85]:


#Store the predicted data in 'predicted' array
predicted = model.predict(X_test)
    # Import metrics
from sklearn import metrics
# generate evaluation metrics-
print(metrics.accuracy_score(y_test, predicted))

