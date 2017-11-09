# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:23:32 2017

@author: yashr
"""

#ML algorithms

np.random.seed(0)

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score



error = lambda x,y : (x-y).abs()


test_df['is_train'] = np.random.uniform(0, 1, len(test_df)) <= .8

# Create two new dataframes, one with the training rows, one with the test rows
train, test = test_df[test_df['is_train']==True], test_df[test_df['is_train']==False]

#temp['is_validation'] = np.random.uniform(0, 1, len(temp)) <= .5
#validation, test = temp[temp['is_validation']==True], temp[temp['is_validation']==False]


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
#print('Number of observations in the validation data:',len(validation))
print('Number of observations in the test data:',len(test))

features = test_df.columns[:7]
target = np.asarray(test_df['stats.average'])





# Create a random forest regressor
random_forest = RandomForestRegressor(n_jobs=2, random_state=0)

dt = DecisionTreeRegressor() 
ada_boost = AdaBoostRegressor(n_estimators=110, base_estimator=dt,learning_rate=1)     

gradient_boost = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, 
                                           max_depth=1, min_samples_split=2, loss="lad")



#creating a plot of the accuracies of each algorithms
def getAccuracy(clf):
    return abs(cross_val_score(clf, test_df[features], target, cv=5, scoring='neg_mean_absolute_error').mean())

prediction_data = pd.DataFrame(
        {'Algorithm': ["Random Forest", "AdaBoost", "Gradient Boost"], 
         'Average Distance from Actual Value': [getAccuracy(random_forest),getAccuracy(ada_boost),getAccuracy(gradient_boost)]
        })    
    
prediction_plot = sns.barplot(y='Average Distance from Actual Value', x= 'Algorithm', data=prediction_data)
prediction_plot.set_title('Accuracy of Different Prediction Algorithms')
prediction_plot.get_figure().savefig('plots/PredictionPlot.png',bbox_inches='tight')


    


#Creating a plot of the importances of different features
random_forest.fit(train[features], train['stats.average'])
importance_data = pd.DataFrame(
        {'Importance': random_forest.feature_importances_, 
         'Feature': features
        })
importance_data['Feature'] = importance_data['Feature'].apply(lambda x: x[x.index('.')+1 :])
importance_data['Importance'] = importance_data['Importance'].apply(lambda x: x*100)
importance_data.sort_values(by='Importance', inplace=True)
colors = reversed(sns.color_palette("Greens_d", n_colors=len(importance_data['Feature'])))
#sns.set_style("whitegrid")
sns.set_context('talk')
importance_plot = sns.barplot(y='Feature', x= 'Importance', data=importance_data, palette=colors)
importance_plot.set_title('Board Game Characteristics Ranked By Importance')
importance_plot.get_figure().savefig('plots/ImportancePlot.png',bbox_inches='tight')

    
    
    
    

#making a residual plot for the gradient boost algorithm
residual_target = np.asarray(train['stats.average'])
gradient_boost.fit(train[features], residual_target)
test_preds = gradient_boost.predict(test[features])
residuals = test['stats.average'] - test_preds

residual_plot = sns.residplot(x=test_preds, y=residuals)
residual_plot.set_title('Residuals for Gradient Descent',fontsize=16)
residual_plot.set(xlabel="Average Rating", ylabel="Residual")
residual_plot.get_figure().savefig('plots/ResidualPlot.png',bbox_inches='tight')



    
    
    

"""Notes"""
"""
RMSE vs MAE - RMSE gives more weight to higher errors bc of squaring
            - useful when large errors are particularly undesirable
            - the more distributed teh errors are, teh higher RMSE will be
            - had a tendency to be larger as the test sample size increases <- can be a big problem in real world modeling
RMSLE- "Root Mean Squared Log Error"
     - used when you dont want to penalize huge differences between predicted and actual values
       when the values themselves are huge
       
Explained Variance - the square ofa the correlation coefficient between x and y
    ex. If the correlation coefficient was 0.89, the explained variance would be 0.8
        This would mean that "X explains 80% of the variance in Y." The other 20% comes from 'Prediction error'

R^2 - Explained Variance/ Total Variance
    - lots of caveats on its use, do more research before using

      

Residual - Error of a prediction
         - should always check residual plots to see how good the fit is 
         - if theres a pattern in the residual plot, then there is probably a way to improve your model


    
""" 
"""using a validation set"""
    # Train the Classifier to take the training features and learn how they relate
    # to the training target
    #clf.fit(train[features], target)
    
    #valid_preds = clf.predict(validation[features])
    #test_preds = clf.predict(test[features])
    
    #valid_error = error(valid_preds, validation['stats.average'])
    #test_error = error(test_preds, test['stats.average'])
    
"""using CV to find optimals hyper-parameters"""
    #scores = []
    #for i in range(2,6):
    #    clf = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, min_samples_split=2)
    #    score = cross_val_score(clf, train[features], target, cv=5)
    #    scores.append(str(score.mean()) + "+/-" + str(score.std()*2))
    #return scores
   
    
    
#unique, counts = numpy.unique(preds, return_counts=True)
#dict(zip(unique, counts))

#clf.predict_proba(test[features])[0:10]
#list(zip(train[features], clf.feature_importances_ ))