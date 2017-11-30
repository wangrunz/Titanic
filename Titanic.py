
# coding: utf-8

# In[158]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

get_ipython().magic('matplotlib inline')


# In[159]:

training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
datasets=[training_data,test_data]


# In[160]:

training_data.head()


# In[161]:

training_data.info()


# In[162]:

test_data.info()


# In[163]:

training_data.describe()


# In[164]:

training_data[['Pclass','Survived']].groupby(['Pclass']).mean()


# In[165]:

training_data[['Sex','Survived']].groupby(['Sex']).mean()


# In[166]:

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


# In[167]:

for dataset in datasets:
    # Mapping Sex to binary
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Fill Embarked missing data to Mode
    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode().item())
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
 
    
    # Fill Fare missing data to Mean
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
    
    # Get avg std and null count of age
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    # Generate random age within 95% confidence interval 
    age_null_random_list = np.random.randint(age_avg - 1.96*age_std/(age_null_count**(0.5)), age_avg + 1.96*age_std/(age_null_count**(0.5)), size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
    # Extract Title from Name, replace french title to uniform title, replace rare titles to rare
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# In[168]:

drop_elements = ['Cabin','Ticket','PassengerId','Name']
training_data = training_data.drop(drop_elements, axis = 1)
test_data = test_data.drop(drop_elements, axis=1)


# In[173]:

test_data.head()


# In[170]:

test_data.info()


# In[171]:

train=training_data.values
X = train[0::, 1::]
Y = train[0::, 0]
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,random_state=1)


# In[189]:

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(3)
fitted_knn=knn.fit(X_train,Y_train)
fitted_knn.score(X_test,Y_test)


# In[190]:

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
fitted_dtc=dtc.fit(X_train,Y_train)
fitted_dtc.score(X_test,Y_test)


# In[191]:

from sklearn.svm import SVC
svc=SVC(probability=True)
fitted_svc=svc.fit(X_train,Y_train)
fitted_svc.score(X_test,Y_test)


# In[ ]:



