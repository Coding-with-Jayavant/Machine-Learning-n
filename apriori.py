# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:46:29 2023

@author: jwark
"""

#apriori
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

#-----------------------read data

df=pd.read_csv(r"D:\DATA SCIENCE\Machine_learning\ML ALGORITHM\unsupervised learning\aproiry\book.csv")
df
#--------------------count the frequency of each item 
#split frequency and item from dict

from collections import Counter
item_frequencies=Counter(df)
item_frequencies

item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies

#when we execute this,items frequencies will be in sorted form 
# item name with count

items=list(reversed([i[0] for i in item_frequencies]))
items
# This is the list comprehenssion it will give the items from dictionaries 

frequencies=list(reversed([i[1] for i in item_frequencies]))
frequencies
#This will  give he frequencies of each items
#----------------Modle building
df

#now apply apriori algorithm on data to calculate the support
frequency_item=apriori(df,min_support=0.0075,max_len=4,use_colnames=True)
frequency_item
# You will get support value for 1,2,3,4 max items

# let us sort the support values
frequency_item.sort_values('support',ascending=False,inplace=True)
frequency_item

# This will sort the support the value in descending order 
# in EDA also there was same trend there it was a count
# and here it was support value

rules=association_rules(frequency_item,metric='lift',min_threshold=1)
# This generate association rule of size 1198x9 columns
# comprises of antescends,consequences
rules.head()
rules.columns
rules.sort_values('lift',ascending=False).head()

###############################################################################################################
##################################################################################################
#
from mlxtend.frequent_patterns import apriori,association_rules

groceries=[]
#with open(r"D:\DATA SCIENCE\recommandation\groceries.csv") as f:groceries=f.read()
f= open(r"D:\DATA SCIENCE\Machine_learning\ML ALGORITHM\unsupervised learning\aproiry\groceries.csv") 
groceries=f.read()
groceries
# it will split data in the comma separater form

groceries=groceries.split('\n')
groceries
#Earlier it was in the format of srin g it will convert it into the form of
# list
groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(','))
groceries_list
# it will separate items from each list so further we can separe it for support caculation
len(groceries_list)
#Out[19]: 9836

all_groceries_list=[i for item in groceries_list for i in item]
all_groceries_list
# we will get all the transactions/item 
# we will get 43368 items in various transaction
len(all_groceries_list)

#-----------------------Now let's count the frequency of each item 
#split frequency and item from dict

from collections import Counter
item_frequencies=Counter(all_groceries_list)
item_frequencies

item_frequencies=sorted(item_frequencies.items(),key=lambda x:x[1])
item_frequencies
#('soda', 1715),('rolls/buns', 1809),('other vegetables', 1903),('whole milk', 2513)]

#when we execute this,items frequencies will be in sorted form 
# item name with count

items=list(reversed([i[0] for i in item_frequencies]))
items
# This is the list comprehenssion it will give the items from dictionaries 

frequencies=list(reversed([i[1] for i in item_frequencies]))
frequencies

#----------------Now we will convert it into dataframe
import pandas as pd
groceries_series=pd.DataFrame(pd.Series(groceries_list))
# Now we will get the the  dataframe of size 9836x1
# the last row of the dataframe is empty so we will remove it
groceries_series=groceries_series.iloc[:9835,:]
groceries_series
groceries_series.head(5)
# So it will remove the last row

# groceries_series having column name 0 so rename as Transaction
groceries_series.columns=['Transactions']
groceries_series

# So there is various elements which is separeted by , = we will seperate using
# * we will join it
x=groceries_series['Transactions'].str.join(sep='*')
x

# Now we will apply one-hot encoding to convert it into numeric form
#to find th co relation
#to convert categorical data into numerical
x=x.str.get_dummies(sep='*')
x

# This is the data which we are going to apply for the Apriori algorithm
#Support is the number of actual occurrences of the data in the specified dataset.
#in frq we only calculate support
frequency_items=apriori(x,min_support=0.0075,max_len=4,use_colnames=True)
frequency_items
# You will get support value for 1,2,3,4 max items

# let us sort the support values
frequency_items.sort_values('support',ascending=False,inplace=True)
frequency_items

# This will sort the support the value in descending order 
# in EDA also there was same trend there it was a count
# and here it was support value

rules=association_rules(frequency_items,metric='lift',min_threshold=3)
# This generate association rule of size 1198x9 columns
# comprises of antescends,consequences
rules.head(20)
rules.columns
rules.sort_values('lift',ascending=False).head(10)