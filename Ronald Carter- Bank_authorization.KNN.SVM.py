#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name: Ronald Carter 
# Purpose: ML Final Project
# Term: Fall 2023- Data 203 Dr.Edmund 


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)


# In[2]:


# read in csv
file_path = '/users/mariocart/Desktop/Projects/ML Folder/Banknote.csv'
bank_df = pd.read_csv(file_path)
bank_df


# In[3]:


bank_df.columns


# ## Feature analysis 

# In[4]:


sns.scatterplot(x='variance',y='skewness',hue='class',data=bank_df,alpha=0.7)


# In[5]:


sns.scatterplot(x='variance',y='curtosis',hue='class',data=bank_df,alpha=0.7)


# In[6]:


sns.scatterplot(x='variance',y='entropy',hue='class',data=bank_df,alpha=0.7)


# In[7]:


sns.scatterplot(x='skewness',y='curtosis',hue='class',data=bank_df,alpha=0.7)


# In[8]:


sns.scatterplot(x='skewness',y='entropy',hue='class',data=bank_df,alpha=0.7)


# In[9]:


sns.scatterplot(x='curtosis',y='entropy',hue='class',data=bank_df,alpha=0.7)


# In[10]:


plt.figure(figsize=(8,6))
sns.heatmap(bank_df.corr(),cmap='coolwarm')


# ## Insights: 
# - Variance and skewness are the features that have the strongest coorlation to class
# - some features that have strong coorlations to each other are: 
# - Skewness and Entrophy
# - Skewness and Curtosis 

# In[11]:


# import splitting,scaling, pipeline, gridsearch, metrics, and model libraries
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[12]:


# defining features and target variables 
X = bank_df.drop(['class'],axis=1)
y = bank_df['class']


# In[13]:


# inspect X & y variables 
X


# In[14]:


y


# In[15]:


# splitting the data into x train & test and y train and test 
# test size .3(30 percent of the data will be for testing and 70 percent will be for training)
# random state basically saves the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)


# # Pipeline Creation 
# - Creating a pipeline to optimize training and scaling 

# In[16]:


# initializing scaler and classifier 
scaler = StandardScaler()
knn = KNeighborsClassifier()


# In[17]:


# defining operations( creates a list of tuples saved under operations that now contains the preprocessing and the classifier
operations = [('scaler',scaler),('knn',knn)]


# In[18]:


# defining pipeline as the operations 
pipe = Pipeline(operations)


# # Implementation of GridSearch

# In[19]:


# defining range( how many k neighbors to test for each iternation)
k_values = list(range(1,30))
# gets a list of 1- 29
k_values


# ## Hyperparameter tuning for the K-Nearest Neighbors (KNN) classifier 
# - using cross-validation and grid search to find the k value that produces best fitting model

# In[20]:


# creating a parameter grid for gridsearchcv which specifies the hyperparameters to search over. 
# basically importing the list of 1-29 to parameter grid for 
param_grid = {'knn__n_neighbors': k_values}


# In[21]:


# basically creating a framework to systematically explore various combinations of our defined parameters(various ks)
# cv is set to five meaning it will divide dataset into 5 parts to train and evaluate the model
# scoring is set to accuraracy to evaluate performance  

full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')


# In[22]:


# basically fits entire framework to training data (training, validation same thing )
full_cv_classifier.fit(X_train,y_train)


# 
# - After fitting, this retrieves the parameters of the best-performing estimator (best model) found by grid search. - returns a dictionary containing the hyperparameters and their corresponding values for the best model.

# In[23]:


full_cv_classifier.best_estimator_.get_params()


# ## Use the output dictionary creating a plot of the mean test scores per K value.

# In[24]:


full_cv_classifier.cv_results_['mean_test_score']


# In[25]:


scores = full_cv_classifier.cv_results_['mean_test_score']
plt.plot(k_values,scores,'o-')
plt.xlabel("K")
plt.ylabel("Accuracy")


# # Final Model evaluation 
# - Using new found insights from cross validation and grid search, evaluationing best model 

# In[26]:


# using best fitting model found from the framework predict on testing data (data the model has not seen yet)
pred = full_cv_classifier.predict(X_test)


# In[27]:


# evaluate performance comparing predications to the actual true labels
confusion_matrix(y_test,pred)


# ## Insights from array
# - 26 instances were correctly predicted as positive.
# - 185 instances were correctly predicted as negative.
# - 1 instance was incorrectly predicted as positive when it was actually negative.
# - 0 instances were incorrectly predicted as negative when they were actually positive.

# In[28]:


print(classification_report(y_test,pred))


# ## This is a great fitting model 

# ## Now for Support Vector Machines 
# - Support Vecton Machines basically try and divide the data using a hyperplane 
# - the hyperplane is fit in the data kind of like a regression line except you want the data points on each side of the line/plane instead of on the plane/line.
# - the goal is to you the plan to classify the points with all of one caterogy on one side of the plane and the other catergy on the other side 
# - their might be a few points on the plan however and the data points that are on the plane are the support vectors 

# In[29]:


# Import neccessary libraries 
import pandas as pd # load and data manipulation 
import numpy as np # data manipulation 
import matplotlib.pyplot as plt # for drawing graphs 
import matplotlib.colors # for more colors in graph 
from sklearn.utils import resample # downsample the dataset 
from sklearn.model_selection import train_test_split # fro data split 
from sklearn.preprocessing import scale #scale and center data 
from sklearn.svm import SVC # the actual model
from sklearn.model_selection import GridSearchCV #for cross validation 
from sklearn.metrics import confusion_matrix # this creates a confusion matrix 
from sklearn.decomposition import PCA # collapsing columns 
import plotly.express as px


# In[30]:


# create a copy of dataframe
bank_copy4= bank_df.copy()


# In[31]:


bank_copy4.columns


# In[32]:


# visualze all four features together 
sns.scatterplot(x='variance', y='skewness', hue='class', size='entropy',
                style='curtosis', data=bank_copy4, palette='Set1', legend=False)


# In[33]:


# better see the graph in 3 dimensions using all 4 features and class target
fig = px.scatter_3d(bank_copy4, x="variance", y="skewness", z="curtosis", color="class", symbol="entropy")
fig.show()


# In[34]:


# get the length of the data 
len(bank_copy4)


# In[35]:


# define features and target 
X4 = bank_copy4.drop(['class'], axis=1)
y4 = bank_copy4['class']


# In[36]:


len(X4)


# In[37]:


len(y4)


# ## Format and Scale the Data
# 

# In[38]:


# scale data so that each point has mean of 0 and std of 1 of close to those
X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.3, random_state=87)
X_train_scaled4 = scale(X_train4)
X_test_scaled4 = scale(X_test4)


# # Initialize preliminary SVM

# In[39]:


clf_svm4 = SVC (random_state = 70)
clf_svm4.fit(X_train_scaled4, y_train4)


# ## Now that the Support vector machine for classification is built lets see how it performs on testing data.

# In[40]:


# making predictions on the testing data
y_pred_test = clf_svm4.predict(X_test_scaled4)

# evaluating the model's performance on testing data (using accuracy as an example)
accuracy = accuracy_score(y_test4, y_pred_test)
print(f"Accuracy on testing data: {accuracy:.4f}")


# # Got 100 percent accuracy so no need to do anything further
# - if i didnt get this I would implenment grid search cross validation to deterine best parameters for best fitting model.
# - Presentation purposes: talk about Scree Plot and PCA
# 

# In[41]:


pca = PCA() # NOTE: By default, PCA() centers the data, but does not scale it.
X_train_pca = pca.fit_transform(X_train_scaled4)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range (1, len (per_var)+1), height=per_var) 
plt.tick_params( 
    axis='x', # changes apply to the x-axis
    which= 'both',#both maior and minor ticks are affected
    bottom=False,#ticks along the bottom edge are off
    top=False,
# ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of 
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Scree Plots')
plt.show()


# ## Scree plot shows how good the approximatation of the true classifer is 
# 
# 

# In[42]:


train_pc1_coords = X_train_pcal[:, 0]
train_pc2_coords = X_train_pcal[:, 1]


pca_train_scaled = scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

param_grid = [
{'C': [1, 10, 100, 1000,
'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
'kernel': ['rbf'1]},
]
optimal_params
( this too much right now come back )

