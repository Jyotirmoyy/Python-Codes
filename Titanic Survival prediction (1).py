#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Importing data from seaboarn

df = sns.load_dataset('titanic')


# In[3]:


# First 10 rows of the data set 

df.head(10)


# In[4]:


# No of rows and columns

df.shape


# In[5]:


df.describe()


# In[6]:


# Counting the survival's number

df['survived'].value_counts()


# In[7]:


# Visulising number of survivals

sns.countplot(x = df['survived'])
plt.show()


# In[8]:


# Visulising columns: 'who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked'

cols = ['who', 'sex', 'pclass', 'sibsp', 'parch', 'embarked']

n_rows  = 2
n_cols = 3

fig, axs = plt.subplots(n_rows, n_cols, figsize = (15,10))

for r in range(n_rows):
    for c in range(n_cols):
        
        i = r*n_cols + c  # Index to go throgh each columns
        ax = axs[r][c]    # Show where to position each sub plots 
        sns.countplot(x = df[cols[i]], hue = df['survived'], ax = ax)
        ax.set_title(cols[i])
        ax.legend(title = 'survived', loc = 'upper right')
plt.tight_layout()


# In[9]:


# Survival rate on the basis of gender 

df.groupby('sex').mean()['survived']*100 


# #### Hence female has ~ 75% chance of survival while male has only ~19%

# In[10]:


# Survival rate on the basis of gender and class 
# Method 1

df.groupby(['sex','class']).mean()['survived']


# In[11]:


# Method 2

df.pivot_table('survived', index = 'sex', columns = 'class')


# #### Hence female having first class are more likely to survived and male having third class are more likely to not survived

# In[12]:


# Survival rate on the basis of gender and class visually 

df.pivot_table('survived', index = 'sex', columns = 'class').plot()
plt.show()


# In[13]:


sns.barplot(x = 'class', y = 'survived', data = df)
plt.show()


# In[14]:


# Survival rate on the basis of age, gender & class

age = pd.cut(df['age'], [0,18,80])

df.pivot_table('survived', ['sex', age], 'class')


# In[15]:


# Price paid for each class

plt.scatter(df['fare'], df['class'], color = 'purple', label = 'Passenger paid ')
plt.xlabel('Price / Fare')
plt.ylabel('Class')
plt.title('Price for each')
plt.legend()
plt.show()


# In[16]:


# number of missing vakue of each columns 

df.isna().sum()


# In[17]:


# Count the values of each columns 

for val in df:
    print(val.upper())
    print()
    print(df[val].value_counts())
    print('\n')
    print()


# In[18]:


# Removing rows of missing values 

df = df.dropna(subset = ['age', 'embarked'])


# In[19]:


# Droping some coreated columns 

df = df.drop(['deck', 'embark_town', 'alone', 'who', 'adult_male', 'alive','class'], axis = 1)


# In[20]:


df.shape


# In[21]:


df.dtypes


# In[22]:


# Printing the unique value in the columns 


print(df['sex'].unique())
print(df['embarked'].unique())


# In[23]:


from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

# Encoding the gender column
df.iloc[:,2] = labelencoder.fit_transform(df.iloc[:,2].values)

# Encoding the embarked column
df.iloc[:,7] = labelencoder.fit_transform(df.iloc[:,7].values)


# In[24]:


# Encoded values of the columns
print(df['sex'].unique())
print(df['embarked'].unique())


# In[25]:


df.dtypes


# In[26]:


# Split the data into independent 'X' and dependent 'Y'

X = df.drop('survived', axis = 1).values
Y = df['survived'].values


# In[27]:


# Scaled the data to avoid some variable to dominates

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)


# In[28]:


# Spliting the data into train test split

from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)


# In[30]:


# Creat a function with many machine leraning models 

def models(X_train, Y_train):
    
    #Logistic regression 
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver = 'lbfgs')
    log.fit(X_train,Y_train)
    
    # Use KNeighbors 
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors= 5, metric = 'minkowski', p =2)
    knn.fit(X_train,Y_train)
    
    #Use SVC (linear model)
    from sklearn.svm import SVC
    svc_lin = SVC(kernel = 'linear', random_state = 0, gamma = 'auto')
    svc_lin.fit(X_train,Y_train)
    
    # Use SVc (RBF Kernal)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel = 'rbf', random_state = 0, gamma = 'auto')
    svc_rbf.fit(X_train,Y_train)
    
    # Use gaussianNB
    from sklearn.naive_bayes import GaussianNB
    guss = GaussianNB()
    guss.fit(X_train,Y_train)
    
    # Use decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train,Y_train)
    
    # Use Random forest 
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=  10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train,Y_train)
    
    
    # Print the training accuracy score
    print('[0] Logistic Regression Score : ', log.score(X_train,Y_train))
    print('[1] K Neighbors Score : ', knn.score(X_train,Y_train))
    print('[2] SVM (Linear) Score : ', svc_lin.score(X_train,Y_train))
    print('[3] SVM (RBF) Score : ', svc_rbf.score(X_train,Y_train))
    print('[4] Guassian NB Score : ', guss.score(X_train,Y_train))
    print('[5] Decision tree : ', tree.score(X_train,Y_train))
    print('[6] Rndom forest :', forest.score(X_train,Y_train))
    
    
    return log, knn, svc_lin, svc_rbf, guss, tree, forest


# In[31]:


models = models(X_train,Y_train)


# In[32]:


from sklearn.metrics import confusion_matrix

for i in range(len(models)):
    cm = confusion_matrix(Y_test, models[i].predict(X_test))
    
    # Extract TN, FP, FN, TP 
    TN, FP, FN, TP = confusion_matrix(Y_test, models[i].predict(X_test)).ravel()
    
    test_score = ( TP +TN ) / ( TP + TN + FP + FN )
    
    print(cm)
    print('Model [{}] Testing Accuracy = "{}"'.format(i, test_score))
    print()


# In[33]:


# Get features importance 
forest = models[6]
importances = pd.DataFrame({'feature': df.iloc[:,1:8].columns, 'importance': np.round(forest.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending = False).set_index('feature')
importances


# In[34]:


importances.plot.bar()
plt.show()


# In[35]:


pred = models[6].predict(X_test)
print(pred)
print()
print(Y_test)


# In[44]:


# My survival
my_survival = [[0,  0, 21, 1, 1, 900, 0]]

# Scaling my survival
my_survival_scaled = sc.fit_transform(my_survival)

# Printing my survival using random forest calssifier

pred = models[6].predict(my_survival_scaled)
print(pred)


if pred == 0:
    print( 'Oh No ! You did not make it')
else:
    print('Nice ! You are survived ')


# In[ ]:




