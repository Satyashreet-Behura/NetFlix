#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')




# In[3]:


df=pd.read_csv("C:\\Users\\satya\\OneDrive\\for power bi\\netflix_titles.csv")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[10]:


df.shape


# In[11]:


df.columns


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum().sort_values(ascending=False)


# In[6]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[7]:


df["director"].value_counts().head(10) # univariate analysis


# # Movies Vs TVShows

# In[17]:


df.type.value_counts()


# In[9]:


a=np.array([6131,2676])
mylabels = ["Movie", "TV Show"]
mycolors=["black","red"]
plt.pie(a, labels = mylabels,colors=mycolors)
plt.legend()


# In[16]:


df.rating.value_counts()


# In[67]:


sns.barplot(x=df.rating.value_counts(),y=df.rating.value_counts().index)
plt.show()


# The highest count - TV-MA is the rating that shows that a program is intended for adults.'MA' stands for "mature audiences". Children aged 17 and younger should not view these programs.
# 
# Second largest is the 'TV-14'. A TV-14 program is meant for children over 14 years of age.It is generally not recommended to let children watch the program without parental attendance, or at least without them vetting it first. It can contain crude humor, the use of harmful substances, strong language, violence and complex or upsetting themes.
# 
# Third largest is the very popular 'R' rating. R is the sort for retricted, so any young person under 17 should not watch.

# In[18]:


df["country"].value_counts().head(10)


# # Year Wise Count
# 

# In[74]:


plt.figure(figsize=(12,10))
ax=sns.countplot(y='release_year',data=df,order=df.release_year.value_counts().index[0:15])


# Highest releases are in 2018 which is followed by 2017 and 2019

# In[23]:


# top 10 directors

df["director"].value_counts().head(10) 


# In[24]:


df["listed_in"].value_counts().head(10)


# In[26]:


plt.figure(figsize=(12,10))
ax=sns.countplot(y='listed_in',data=df,order=df.listed_in.value_counts().index[0:15])


# # Handling Missing Values

# In[29]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[30]:


round(df.isnull().sum())


# In[35]:


# Droping rows for small percentage of null value

df.dropna(subset=['rating','duration','date_added'],axis=0,inplace=True)


# In[36]:


df.shape  # after removal


# In[37]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[38]:


# replacing missing values in country with "unknown"

df.country.replace(np.NaN,"unknown",inplace=True)


# In[39]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[40]:


# replacing missing values in Cast with "No Cast"

df.cast.replace(np.NaN,"No Cast",inplace=True)


# In[41]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# In[42]:


df["country"].value_counts().head(10)


# In[43]:


df["cast"].value_counts().head(10)


# In[44]:


# replacing missing values in director with "No director"

df.director.replace(np.NaN,"No director",inplace=True)


# In[45]:


round(df.isnull().sum()/df.shape[0]*100,2).sort_values(ascending=False)


# it is clear now

# In[46]:


df.title


# In[33]:


cast_shows=df[df.cast !='No Cast'].set_index('title').cast.str.split(',', expand=True).stack().reset_index(level=1, drop=True)
plt.figure(figsize=(13,7))
plt.title(' Top 10 actor movies based on the no. of titles')
sns.countplot(y =cast_shows, order=cast_shows.value_counts().index[:10])
plt.show()


# In[39]:


movies_df=df.loc[(df['type']=='Movie')]
movies_df.head(2)


# In[25]:


show_df=df.loc[(df['type']=='TV Show')]
show_df.head(2)


# In[46]:


show_df.duration=show_df.duration.apply(lambda x: x.replace(" Season","")if 'Season' in x else x)
show_df.head(2)


# In[48]:


show_df.duration=show_df.duration.apply(lambda x: x.replace("s","")if 's' in x else x)
show_df.head(2)


# In[66]:


show_df.loc[:,["duration"]]=show_df.loc[:,["duration"]].apply(lambda x: x.astype('int64',errors='ignore'))


# In[67]:


show_df.duration.value_counts().tail(10)


# In[58]:


# show with highest no of seasons
longest_shows=show_df.loc[(show_df['duration']>13)]
longest_shows


# In[68]:


longest_shows.rating.value_counts()


# In[45]:


df[['release_year']].head(10)


# In[90]:


x=df.release_year.value_counts().head()


# In[42]:


netflix_date=df[["date_added"]].dropna()
print(netflix_date)

netflix_date["year"]=netflix_date["date_added"].apply(lambda x: x.split(', ')[-1])
print(netflix_date["year"]) # exctracted year column

netflix_date["month"]=netflix_date["date_added"].apply(lambda x: x.split(' ')[0])
print(netflix_date["month"]) # exctracted month column

print(netflix_date) # final dataframe


# In[30]:


month_order= ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
new_df=netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T
print(new_df)          # pivoted table


# In[72]:


# plotting the heatmap

plt.figure(figsize=(10,7))
plt.pcolor(new_df,cmap='Greens',edgecolors='white',linewidths=2)
plt.xticks(np.arange(0.5,len(new_df.columns),1),new_df.columns, fontsize=10, fontfamily='calibri')
plt.yticks(np.arange(0.5,len(new_df.index),1),new_df.index,fontsize=10, fontfamily='calibri' )

plt.title("Months Vs Netflix Contents Update",fontsize=22, fontfamily='calibri',fontweight='bold')
cbar=plt.colorbar()


# In[ ]:




