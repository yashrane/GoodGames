# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:10:37 2017

@author: yashr
"""






import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
np.random.seed(0)



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
    
    round_col = np.vectorize(lambda x: int(round(x)))
    test_df = test_df.apply(round_col, axis='columns')
    has_score = test_df['stats.average'] != 0
    test_df = test_df[has_score]
    return test_df
    
df = loadData()
test_df = loadTestData(df)


    
test_df['is_train'] = np.random.uniform(0, 1, len(test_df)) <= .6

# Create two new dataframes, one with the training rows, one with the test rows
train, temp = test_df[test_df['is_train']==True], test_df[test_df['is_train']==False]

temp['is_validation'] = np.random.uniform(0, 1, len(temp)) <= .5
validation, test = temp[temp['is_validation']==True], temp[temp['is_validation']==False]


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the validation data:',len(validation))
print('Number of observations in the test data:',len(test))

features = test_df.columns[:7]
target = np.asarray(train['stats.average'])

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training target
clf.fit(train[features], target)

valid_preds = clf.predict(validation[features])
test_preds = clf.predict(test[features])



error = lambda x,y : (x-y).abs()
valid_error = error(valid_preds, validation['stats.average'])
test_error = error(test_preds, test['stats.average'])



#unique, counts = numpy.unique(preds, return_counts=True)
#dict(zip(unique, counts))

#clf.predict_proba(test[features])[0:10]
#list(zip(train[features], clf.feature_importances_))











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
