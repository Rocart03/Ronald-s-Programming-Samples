#!/usr/bin/env python
# coding: utf-8

# ## Decision Trees vs. Random Forest

# In[ ]:


# Name: Ronald Carter 
# Purpose: ML Final Project
# Term: Fall 2023- Data 203 Dr.Edmund 


# In[1]:


# Import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# read in csv
file_path = '/users/mariocart/Desktop/Projects/ML Folder/Banknote.csv'
bank_df = pd.read_csv(file_path)
bank_df.head()


# In[3]:


bank_df.columns


# 

# ## Train, Test, Split

# In[4]:


# defining features and target variables 
X = bank_df.drop(['class'],axis=1)
y = bank_df['class']


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Decision Tree Classifier
# 
# ## Default Hyperparameters

# In[7]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[8]:


model.fit(X_train,y_train)


# In[9]:


base_pred = model.predict(X_test)


# In[12]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,base_pred)


# In[13]:


import matplotlib.pyplot as plt
import numpy as np

# confusion matrix values
conf_matrix = np.array([[231, 7],
                        [1, 173]])

# plotting the confusion matrix
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()

# adding labels and ticks
classes = ['Class 0', 'Class 1']  # Replace with your class names if needed
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# adding annotations
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             ha="center", va="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()


# - True Positive (TP): 173 instances were correctly predicted as Class 1.
# - True Negative (TN): 231 instances were correctly predicted as Class 0.
# - False Positive (FP): 7 instances were predicted as Class 1 but actually belong to Class 0.
# - False Negative (FN): 1 instance was predicted as Class 0 but actually belongs to Class 1.

# In[14]:


print(classification_report(y_test,base_pred))


# ## Precision:
# 
# - Class 0: When the model predicts an instance as Class 0, it is correct 100% of the time.
# - Class 1: When the model predicts an instance as Class 1, it is correct about 96% of the time.
#  
# 
# ## Accuracy:  98%, model correctly predicts 98% of the instances in the test dataset.
# 

# In[15]:


model.feature_importances_
pd.DataFrame(index=X.columns,data=model.feature_importances_,columns=['Feature Importance'])


# - variance has the highest importance (0.636716).
# - skewness is the second most important (0.204654).
# - curtosis lower importance (0.132978).
# - entropy has the lowest importance (0.025652).

# In[16]:


# viz the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12,8))
plot_tree(model);


# In[17]:


# decision tress with class colors to make easier to comprehend 
plt.figure(figsize=(12,8),dpi=150)
plot_tree(model,filled=True,feature_names=X.columns);


# In[18]:


# reporting model results
def report_model(model):
    model_preds = model.predict(X_test)
    print(classification_report(y_test,model_preds))
    print('\n')
    plt.figure(figsize=(12,8),dpi=150)
    plot_tree(model,filled=True,feature_names=X.columns);


# In[21]:


pruned_tree = DecisionTreeClassifier(max_leaf_nodes=2)
pruned_tree.fit(X_train,y_train)


# In[22]:


report_model(pruned_tree)


# In[23]:


from sklearn.model_selection import GridSearchCV # used for hyperparameter tuning using cross validation 
#
n_estimators=[64,100,128,200]
max_features= [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]


# 
# - n_estimators: the number of trees in the forest 
# - max_features: the maximum number of features considered for splitting a node
# - bootstrap: whether to use bootstrapping when building trees, and 
# - oob_score: whether to use out-of-bag samples to estimate the generalization accuracy.
# 

# In[26]:


# param_grid: specifies the grid of hyperparameters to search

param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}  


# In[27]:


from sklearn.ensemble import RandomForestClassifier
 # initializes a Random Forest Classifier with default parameters.
rfc = RandomForestClassifier()
grid = GridSearchCV(rfc,param_grid)


# In[28]:


# perform cross-validation for each combination of hyperparameters defined in the param_grid. (mutiple iterates)
grid.fit(X_train,y_train)


# In[29]:


# get the best combination of parameters 
grid.best_params_


# In[30]:


# uses best model with best parameter combinations on test data
predictions = grid.predict(X_test)


# In[31]:


print(classification_report(y_test,predictions))


# In[32]:


# reporting back original oob_score parameter
grid.best_estimator_.oob_score


# In[33]:


# reporting back fitted attribute of oob_score
grid.best_estimator_.oob_score_


# ## high accuracy of out of bag samples which is just more unseen data used to train model.
