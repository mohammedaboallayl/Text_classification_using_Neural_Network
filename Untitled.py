#!/usr/bin/env python
# coding: utf-8

# <h1>Importing Libraries</h1>

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import joblib
import re


# <h1> Loading Data</h1>

# In[2]:


#loading data
dir = os.path.join(os.getcwd(), "Emails.tsv") #preparing path for DataSet
EmailsData=pd.read_table(dir,header=None, names=['Type', 'Message'])
EmailsData.head()


# <h2> Data Understanding and Preprocessing</h2>

# In[3]:


#display information to ensure that data is complete and dont have any None values
EmailsData.info()


# In[4]:


#viewing count of items in classes
plt.figure(figsize=(20,4))
sns.countplot(x="Type",data=EmailsData)


# In[5]:


# functions for preparing data
def Type(item):#maping type column to 1 for hame and 0 for spa,
    if item=="ham":
        return 1
    else:
        return 0
def Message(item):#deleting any number as it is not relative
    return re.sub(r"[\d]","",item)


# In[6]:


# Applying functions for preparing data
EmailsData["Type"]=EmailsData["Type"].apply(Type)
EmailsData["Message"]=EmailsData["Message"].apply(Message)
EmailsData


# In[7]:


#Splitting data
X_train, X_test, y_train, y_test = train_test_split(np.array(EmailsData["Message"]), np.array(EmailsData["Type"]), test_size=0.3,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[8]:


# Making maping from phrase to vactor of numbers for training
Vcreator= CountVectorizer()
Vcreator.fit(X_train)
X_train_vector = Vcreator.transform(X_train).toarray()
X_test_vector=Vcreator.transform(X_test).toarray()
y_test=np.array(y_test).reshape((y_test.shape[0],1))
y_train=np.array(y_train).reshape((y_train.shape[0],1))


# In[9]:


X_train_vector.shape


# <h1>Creating Neural Network Model</h1>

# <pre>
# <h3> Neural Network Description </h3>
#   1- Don't have any Layer So its not A DNN and that for making learn faster irrespective of making good accuracy 
#   2- Have Only One Output as we are Using Binary Classification
#   3- It's Weights have been assigned to valuse between 0 and 0.1 which make it Faster 
#   4- Have two main function which forward which make forward propagation and back which make backward propagation 
#   5- Have another two function one for making prediction and anoter for training
# </pre>
# <img src="NN.png"/>

# In[20]:


class nuralnetwork:
    def __init__(self,x,y):
        self.X=x
        self.Y=y
        self.W1=np.random.rand(x.shape[1],1)/9 #for Weights of first Layer(input Layer)
        self.threshold=0.5 #for threshold
        self.LearningRate=0.001 #for Learning rate
    def sig(self,x):#sigmoid function
        return 1/(1+np.exp(-x))

    def sig_dev(self,x): #derivative of sigmoid function
        return x*(1.0-x)
    def forward (self):#Forward Propagation 
        self.output = self.sig(np.dot(self.X, self.W1))
        return np.vectorize(self.makethreshold)(self.output)
    def back(self):#backward propagetion
        self.o_d2=(self.Y-self.output)
        self.dw1=self.W1+self.LearningRate*np.dot(self.X.T,(self.o_d2*(self.sig_dev(self.output))))
        self.W1=self.dw1
    def train(self,iterations):# Train function 
        for i in range(iterations):
            self.forward()
            self.back()
    def makethreshold(self,item):
        if item > self.threshold:
            return 1
        else:
            return 0
    def predict(self,data):
        self.output = self.sig(np.dot(data, self.W1))
        return  np.vectorize(self.makethreshold)(self.output)


# <h1 >Creating Object of Neural Network and Train it</h1>

# In[21]:


Train_Error=[]
Test_Error=[]
NN=nuralnetwork(X_train_vector,y_train)
for i in range(100):
    NN.train(10)
    Train_Error.append(accuracy_score(y_train, NN.forward()))
    Test_Error.append(accuracy_score(y_test,NN.predict(X_test_vector)))


# In[22]:


plt.plot(np.linspace(0,99,100),Train_Error,label="Train")
plt.plot(np.linspace(0,99,100),Test_Error,label="Test")
plt.legend()
plt.show()


# <h4> No Overfit as Accuracy of Tarin about 98% and in Test 97%</h4>

# In[23]:


y_pred=NN.predict(X_test_vector)


# In[24]:


metric=confusion_matrix(y_test,y_pred)
print(metric)


# In[25]:


Report=classification_report(y_test,y_pred)
print(Report)


# In[26]:


sns.heatmap(metric,center=True)


# <h1> Saving Model</h1>

# In[27]:


#saving Model
joblib.dump(NN , 'mode.sav')

