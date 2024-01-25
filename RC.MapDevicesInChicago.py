#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Name: Ronald Carter 
# Purpose: Internet Equity, Geospatial Mapping 


# In[1]:


import pandas as pd
import numpy as np
from shapely.wkt import loads


# In[2]:


deviceid = pd.read_csv('../data/device-to-block-map.csv')
deviceid


# In[3]:


import geopandas as gpd
censusblock = gpd.read_file('../data/CensusBlockTIGER2010.csv')

censusblock.geometry = censusblock['the_geom'].apply(loads)
# censusblock_geo = gpd.GeoDataFrame(data=censusblock, geometry=censusblock['the_geom'])


# In[4]:


type(censusblock.geometry)


# In[5]:


censusblock


# In[6]:


#import geopandas as gpd
#from shapely import wkt

#censusblock_gpd = gpd.GeoDataFrame(censusblock, geometry='geometry')
censusblock.plot()


# In[7]:


censusblock = censusblock[['GEOID10', 'geometry']]
censusblock


# In[8]:


deviceid.geoid10.dtypes


# In[11]:


censusblock['GEOID10'] = censusblock.GEOID10.astype('int64')
censusblock['GEOID10'].dtypes


# In[10]:


idcensusblock = pd.merge(deviceid, censusblock, left_on='geoid10', right_on='GEOID10', how='left')
idcensusblock = idcensusblock.drop(['Unnamed: 0', 'GEOID10'], axis=1)
idcensusblock


# In[12]:


idcensusblock_gpd = gpd.GeoDataFrame(idcensusblock, geometry='geometry')
idcensusblock_gpd.plot(color = 'red')


# In[13]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
censusblock.plot(ax=ax)
idcensusblock_gpd.plot(ax=ax, color='red')

