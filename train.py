
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("Manual-Data/Training.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[6]:


len(data.columns)


# In[7]:


len(data['prognosis'].unique())


# In[8]:


df = pd.DataFrame(data)


# In[9]:


df.head()


# In[10]:


cols = df.columns


# In[11]:


cols = cols[:-1]


# In[12]:


cols


# In[13]:


len(cols)


# In[14]:


x = df[cols]
y = df['prognosis']
#print x[:5]
#print y[:5]


# In[15]:


import os
import csv
with open('/home/shashank/Desktop/Predicting-Diseases-From-Symptoms-master/Manual-Data/Training.csv') as f:
    reader = csv.reader(f)
    i = reader.next()
    rest = [row for row in reader]
column_headings = i


# In[16]:


for ix in i:
    ix = ix.replace('_',' ')
    #print ix


# In[17]:


import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[19]:


mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)


# In[20]:


mnb.score(x_test, y_test)


# In[21]:


from sklearn import cross_validation
print ("cross result========")
scores = cross_validation.cross_val_score(mnb, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())


# In[22]:


test_data = pd.read_csv("Manual-Data/Testing.csv")


# In[23]:


test_data.head()


# In[24]:


testx = test_data[cols]
testy = test_data['prognosis']


# In[25]:


mnb.score(testx, testy)


# In[26]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[28]:


#print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)
#print ("Acurracy: ", clf_dt.score(x_test,y_test))


# In[29]:


from sklearn import cross_validation
#print ("cross result========")
scores = cross_validation.cross_val_score(dt, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())


# In[30]:


#print ("Acurracy on the actual test data: ", clf_dt.score(testx,testy))


# In[31]:


#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")


# In[32]:


features = cols


# In[33]:


for f in range(5):
    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))
    


# In[34]:


feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i


# In[35]:


feature_dict['hip_joint_pain']


# In[36]:


sample_x = [i/79 if i==79 else i*0 for i in range(len(features))]


# In[37]:


len(sample_x)


# In[38]:


sample_x = np.array(sample_x).reshape(1,len(sample_x))


# In[39]:


#print dt.predict(sample_x)


# In[40]:
import pickle

decision_tree_pkl_filename = 'decision_tree_classifier.pickle'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'w')
pickle.dump(dt, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()
#print dt.predict_proba(sample_x)

