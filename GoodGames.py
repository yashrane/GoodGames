# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:10:37 2017

@author: yashr
"""






import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns









def loadData():
    #load the dataset from the sqlite file
    cnx = sqlite3.connect('lib/database.sqlite')
    raw_data = pd.read_sql_query("SELECT * FROM BoardGames", cnx)
    
    #drop any columns with NA
    df = raw_data.dropna(thresh=100, axis='columns')
    return df

def loadTestData(df):
    #selecting the specific data i want
  #  test_df = pd.DataFrame(df.iloc[:, 2])
    test_df = pd.DataFrame(df.iloc[:, 5:10])
    test_df = test_df.join(df.iloc[:, 11])
    test_df = test_df.join(df.iloc[:, 13])
    test_df = test_df.join(df['stats.average'])
        
    test_df = test_df.dropna(axis=0, how='any')
    
    #round_col = np.vectorize(lambda x: int(round(x)))
    #test_df = test_df.apply(round_col, axis='columns')
    has_score = test_df['stats.average'] != 0
    test_df = test_df[has_score]
    return test_df
    
df = loadData()
test_df = loadTestData(df)


    




















"""
Notes for possibly useful functions:
df.sample(n) - gets a random sample of n rows from df
df.sort_values(by='column_name', ascending=BOOL)
data['delayed'] = data['arr_delay'].apply(lambda x: x > 0)

data['delayed'].value_counts() - gives the counts for each possible value
df.unstack() - make more readable?
for c1,c2 in df.iteritems(): - for when im really desperate

df.fillna(value='replaced',axis='index/columns', inplace=True )
pd.crosstab()


"""



"""
is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
                                      is_numeric_dtype(df['B'])

is_number(df.dtypes)

df.select_dtypes(include=[np.number])

count_na = np.vectorize(lambda x: x.isnull().sum)
"""
