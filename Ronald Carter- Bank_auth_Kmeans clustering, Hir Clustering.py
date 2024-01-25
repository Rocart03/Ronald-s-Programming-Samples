#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Ronald Carter 
# Date: 10/12/2023
# Machine Learning Bias & Ethic

# import libraries  
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression


# ## Kmeans clustering is an interative algortihm meaning you go through mutiple interatives to update alg and get to the best fitting model 

# In[2]:


# read in csv
file_path = '/users/mariocart/Desktop/Projects/ML Folder/Banknote.csv'
bank_df = pd.read_csv(file_path)
bank_df.head()


# In[3]:


bankdf_copy1=bank_df.copy()
bankdf_copy2=bank_df.copy()


# In[4]:


# inspect df info 
bank_df.info()


# In[5]:


bank_df.describe().transpose()


# In[6]:


# check NAN values 
bank_df.isnull().sum()


# ## Exploratory Data Analysis 

# In[7]:


# Creating a coorlation matrix 
sns.heatmap(bank_df.corr())


# ## Take a closer look into the nature of the relationships between each variable 
# - Specifically looking for obvious patterns and trends 
# 

# In[8]:


# get a lsit of the column names 
cols_names = bank_df.columns
cols_names


# In[9]:


sns.histplot(data=bank_df,x='variance')


# ## The data shows that variance values ranging from -3 to 4.2 have the highest frequency of occurrences.
# - Meaning: In the sample data there are alot of banknotes with grayscale intensity variation between -3 and 4.2.

# In[10]:


sns.histplot(data=bank_df,x='variance',hue='class')


# # Observing the class column intersectionaly with variance column reveals a distinct trend: lower variances predominantly belong to one class, while higher variances align with another. However, there are overlaps in the range between -3 to able 2.4, where both classes variability ranges overlap.
# - Variance is probably a good indicator or class ( good point for later)
# - I will further look into the relationship of other varibles againsts variance.
# 
# 
# 
# 
# 

# In[11]:


# lets visualize some other relationships with variance 
plt.scatter(bank_df.variance, bank_df.skewness)
plt.title('Variance v Skewness')
plt.show()


# In[12]:


plt.scatter(bank_df.variance, bank_df.entropy)
plt.title('Variance v entropy')
plt.show()


# In[13]:


plt.scatter(bank_df.variance, bank_df.curtosis)
plt.title('Variance v curtosis')
plt.show()


# ## So far based on the scatterplots I dont see distrinctive clusters but I did see some what of a pattern when looking at Variance vs. Skewness.

# ## Feature Analysis  with base variable skewness

# In[14]:


sns.histplot(data=bank_df,x='skewness')


# In[15]:


sns.histplot(data=bank_df,x='skewness',hue='class')


# In[16]:


plt.scatter(bank_df.skewness, bank_df.curtosis)
plt.title('Skewness v curtosis')
plt.show()


# In[17]:


plt.scatter(bank_df.skewness, bank_df.entropy)
plt.title('Skewness v Entropy')
plt.show()


# In[18]:


sns.histplot(data=bank_df,x='curtosis')


# In[19]:


sns.histplot(data=bank_df,x='curtosis',hue='class')


# In[20]:


plt.scatter(bank_df.curtosis, bank_df.entropy)
plt.title('Curtosis v Entropy')
plt.show()


# In[21]:


sns.histplot(data=bank_df,x='entropy')


# In[22]:


sns.histplot(data=bank_df,x='entropy',hue='class')


# ## Higher values of skewness are more frequencly one class (values 6-11). extremly low values of skewness are the opposing class (

# In[23]:


# make copy of the dataframe for manipulation 
bank_copy = bank_df.copy()
bank_copy.head()


# In[24]:


# drop the class column being this is unsupervised learning, dont need a label 
bank_copy.drop('class', axis=1, inplace=True)
bank_copy.head()


# In[25]:


# Get statistics of the dataframe ( its funny because it like getting statistics of statistics)
bank_copy.describe().transpose()


# # The columns have different statistical qualities( different means, std, min etc)
# - Data needs to be scaled to normalize. 
# 

# In[26]:


from sklearn.preprocessing import StandardScaler 
import pandas as pd


scaler = StandardScaler()
columns_to_scale = ['variance', 'skewness', 'curtosis','entropy']

# scale the selected columns and replace their values in the DataFrame
bank_copy[columns_to_scale] = scaler.fit_transform(bank_copy[columns_to_scale])

print(bank_copy)


# In[27]:


bank_copy.info()


# In[28]:


# Get stats of dataframe again 
bank_copy.describe().transpose()


# # 
# - Mean: The mean values for all features are very close to zero meaning the data was properly scaled and centered around zero.
# - Std: The standard deviation for each feature is approximately 1.000365 
# - Range: The minimum and maximum values for each feature (min and max) fall within the expected range.

# # Initalize the Centroids 
# 

# In[29]:


data = bank_copy 
def random_centroids(data, k):
    centroids = []
    for i in range(k):
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis = 1)


# In[30]:


centroids = random_centroids(data, 2)


# In[31]:


# so for centroid 0 it would be at that value for variance then skweness so on so on 
centroids


# ## Label each datapoint according to cluster center (centroid)
# - Will evaluate each banknotes distances to each centorid to determine which centroid to label it to.
# - This assignment is determined by calculating the distances between each data point and all centroids
# 

# In[32]:


def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) **2).sum(axis=1)))
    return distances.idxmin(axis=1) 
                                
# assigning cluster labs to labels
                              
                               
                                


# In[33]:


labels = get_labels(data,centroids)  


# In[34]:


labels


# In[35]:


labels.value_counts()


# In[37]:


# wrap into function 
def new_centroids(data, labels, k):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T


# ## Looks like alot of banknotes are in the 0 label 

# ## Checkpoint 
# - Assinged each banknote to a cluster based on the random centroids. 
# - next, updating the centroids based on what banknotes are in each cluster 
# - finding the geometric mean (mutiplying each of them together and taking the nth root based on the numnber of points)

# # Visualize each iteration 
# - iteration is each time the centroid is defined then geometric mean of data points are found and centriod location/distance is updated)

# In[39]:


# import libraries 
# principle component analisys which will essentially turn four dimensal data into two deminensal data.
# will summarize all 4 features into 2 columns 
from sklearn.decomposition import PCA
# for plotting 
import matplotlib.pyplot as plt
# clear graph after each iteration
from IPython.display import clear_output


# In[40]:


# defining function that changes the 4 dimensional features into two for clarify with graphing purposes 
#clears each graph and loads with new one upon each iteration 
#plots the data coloring with each label and plots the centroids each iteration 
def plot_clusters(data, labels, centroids, iteration):
    pca= PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x = data_2d[:,0], y = data_2d[:,1], c = labels)
    plt.scatter(x=centroids_2d[:,0], y = centroids_2d[:,1])
    plt.show()


# ## Put it all together 
# 

# In[41]:


# defining max iterations 
max_iterations = 100
# defining number of clusters 
k = 2
#data = bank_copy

# itenralize centroids 
centroids = random_centroids(data, k)
old_centroids = pd.DataFrame()
iteration = 1


while iteration < max_iterations and not centroids.equals(old_centroids):
    old_centroids = centroids 
    
    labels = get_labels(data, centroids)
    centroids = new_centroids(data, labels, k)
    plot_clusters(data, labels, centroids, iteration)
    iteration += 1
    # the


# ## Hierarchical-Clustering

# In[43]:


# Drop class column from copy of dataframe
bankdf_copy1.drop(columns='class', inplace=True)


# In[45]:


bankdf_copy1.head()


# In[46]:


# import scaler 
from sklearn.preprocessing import MinMaxScaler


# In[47]:


# define scaler 
scaler = MinMaxScaler()


# In[48]:


# fit scaler 
scaled_data1 = scaler.fit_transform(bankdf_copy1)


# In[49]:


scaled_data1


# In[50]:


scaled_df1 = pd.DataFrame(scaled_data1,columns=bankdf_copy1.columns)


# In[52]:


plt.figure(figsize=(15,8))
sns.heatmap(scaled_df1,cmap='magma');


# In[53]:


sns.clustermap(scaled_df1,row_cluster=False)


# In[54]:


sns.clustermap(scaled_df1,col_cluster=False)


# In[55]:


# import agglomerative clustering 
from sklearn.cluster import AgglomerativeClustering


# In[56]:


# define model 
model = AgglomerativeClustering(n_clusters=4)


# In[60]:


# define cluster labels 
cluster_labels1 = model.fit_predict(scaled_df1)


# In[61]:


cluster_labels1


# In[63]:


plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=bankdf_copy1,x='variance',y='skewness',hue=cluster_labels1)


# ## Exploring Number of Clusters with Dendrograms

# In[64]:


# define model
model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)


# In[65]:


cluster_labels1 = model.fit_predict(scaled_df1)


# In[66]:


cluster_labels1


# In[67]:


# imprt dendrograms
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy


# In[69]:


#define linkage matrix 
linkage_matrix = hierarchy.linkage(model.children_)


# In[70]:


linkage_matrix


# In[71]:


plt.figure(figsize=(20,10))
# Warning! This plot will take awhile!!
dn = hierarchy.dendrogram(linkage_matrix)


# In[72]:


plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=48)

