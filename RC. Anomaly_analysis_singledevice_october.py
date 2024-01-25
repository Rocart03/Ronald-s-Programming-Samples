#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Name: Ronald Carter 
# Purpose: Internet Equity Analysis single device, single month 


# In[29]:


#import the entire dataframe from the one device for the month of october 
import pandas as pd
import numpy as np
rc_device = pd.read_csv('/srv/data/my_shared_data_folder/internet-equity/device_data_oct21/nm-mngd-20210518-075ab2f0.csv')
rc_device


# In[30]:


# However the header of the column is an actual row, and should be within the dataframe not the header of the data frame 
# setting the header to none which will put the previous header within the dataframe and then I am going to rename the columns the correct names
import pandas as pd
import numpy as np
rc_device = pd.read_csv('/srv/data/my_shared_data_folder/internet-equity/device_data_oct21/nm-mngd-20210518-075ab2f0.csv', header=None)
rc_device
rc_device.rename(columns={0:'time',1:'deviceid',2:'tool',3:'direction',4:'protocol',5:'target',6:'pktloss',7:'method',8:'zip',9:'isp',10:'value',11:'topic',12:'annonipaddr',13:'ipaddrchanged'},inplace=True)
rc_device


# In[31]:


#seperate the data and hours(why), make each a column and add both as two seperate columns at the end of the dataframe.

def split_hour(x):
    row = x
    return row[11:13]

rc_device.loc[:,"hour"] = rc_device.loc[:,"time"].apply(split_hour)
rc_device


# In[32]:


def split_date(x):
    row = x
    return row[:10]

rc_device.loc[:,"date"] = rc_device.loc[:,"time"].apply(split_date)
rc_device
#now we have two new columns with just the date and just the hour.


# In[33]:


#looking at the target server Atlanta
#making a new dataframe with just atlanta target.
rc_device_atl = rc_device.loc[rc_device['target'] == 'Atlanta']
rc_device_atl


# In[34]:


#making dataframes for just the max, min, avg, and mdev
#This one is for the max
rc_device_atlmax= rc_device_atl.loc[rc_device_atl['method'] == 'max']
rc_device_atlmax


# In[35]:


#This dataframe is for the min
rc_device_atlmin= rc_device_atl.loc[rc_device_atl['method'] == 'min']
rc_device_atlmin


# In[36]:


#This dataframe is for the the avg 
rc_device_atlavg= rc_device_atl.loc[rc_device_atl['method'] == 'avg']
rc_device_atlavg


# In[37]:


#This dataframe is for the mdev
rc_device_atlmdev= rc_device_atl.loc[rc_device_atl['method'] == 'mdev']
rc_device_atlmdev


# In[38]:


#Todays objective require me to work with the max data from the device which i already make a dataframe for called 'rc_device_atlmax'
#I am going to change the column hour into intengers making it easier to work with later on.
#then will print the dtypes to corfirm if it changed.
rc_device_atlmax.hour = rc_device_atlmax['hour'].astype('int')
print(rc_device_atlmax.dtypes)


# In[39]:


#One object today was to look at 7 days from the dataframe.
#Make a random sample of seven days from the atlanta max dataframe rc_device_atlmax
rc_device_atlmax['date'].sample(7, replace=False)



# In[43]:


#Now we have the random sample of seven dates to use. 
#Making a subset for each day, make sure your making using the dataframe rc_device_atlmax
#This one is for the day 10-19
rc_device_atlmax_1019 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-19']
rc_device_atlmax_1019


# In[44]:


#This subset is for 10/12
rc_device_atlmax_1012 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-12']
rc_device_atlmax_1012


# In[45]:


#This subset is for 10/15
rc_device_atlmax_1015 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-15']
rc_device_atlmax_1015


# In[46]:


#This subset is for 10/21
rc_device_atlmax_1021 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-21']
rc_device_atlmax_1021


# In[47]:


#This subset is for 10/01
rc_device_atlmax_1001 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-01']
rc_device_atlmax_1001


# In[48]:


#This subset is should be 10/21 based on the random sample code however being that I already have a subset of that day and the objective was to look at 7 different days Im going to chose a random day to eliminate doing a duplicate day.
#The new date picked is 10/07
rc_device_atlmax_1007 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-07']
rc_device_atlmax_1007


# In[49]:


#This subset is for 10/17
rc_device_atlmax_1017 = rc_device_atlmax.loc[rc_device_atlmax['date'] == '2021-10-17']
rc_device_atlmax_1017


# In[53]:


#Time series plot of the maxes from the date 10/19 (from the one device)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'green', data =rc_device_atlmax_1019,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1019['hour']), max(rc_device_atlmax_1019['hour'])+1, 1.0))


# In[54]:


#Time series plot of the maxes from the date 10/12 (from the one device)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'red', data =rc_device_atlmax_1012,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1012['hour']), max(rc_device_atlmax_1012['hour'])+1, 1.0))


# In[55]:


#Time series plot of the maxes from the date 10/15 (from the one device)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'orange', data =rc_device_atlmax_1015,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1015['hour']), max(rc_device_atlmax_1015['hour'])+1, 1.0))


# In[57]:


#Time series plot of the maxes from the date 10/21 (from the one device)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'purple', data =rc_device_atlmax_1021,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1021['hour']), max(rc_device_atlmax_1021['hour'])+1, 1.0))


# In[64]:


#Time series plot of the maxes from the date 10/01 (from the one device)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'yellow', data =rc_device_atlmax_1021,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1001['hour']), max(rc_device_atlmax_1001['hour'])+1, 1.0))


# In[63]:


#Time series plot of the maxes from the date 10/07 (from the one device) 

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'black', data =rc_device_atlmax_1007,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1007['hour']), max(rc_device_atlmax_1007['hour'])+1, 1.0))


# In[62]:


#Time series plot of the maxes from the date 10/17 (from the one device) 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig,ax =  plt.subplots(figsize = ( 10, 8))
sns.lineplot( x = "hour", y = "value",
             color = 'blue', data =rc_device_atlmax_1007,
             ax = ax)
plt.xticks(np.arange(min(rc_device_atlmax_1017['hour']), max(rc_device_atlmax_1017['hour'])+1, 1.0))


# In[77]:


#Based on the time series plots from the individual day dataframe/subset. I determined an anomaly is probably 42 and greater
#Now creating a function that gos through and identifies an anomaly.
def anomaly_detec(x):
    anomaly_sum=0
    if x >= 42:
        anomaly_sum +=1
    else:
        pass
    return anomaly_sum

    
    


# In[78]:


anomaly_count = rc_device_atlmax_1012['value'].apply(anomaly_detec)
anomaly_count


# In[81]:


anomaly_sum = anomaly_count.sum()
anomaly_sum


# In[ ]:


#There are 67 anomalys in the day 10/12
#now will repeat for all seven days and create a Dictionary to input the day with the anomally count.

