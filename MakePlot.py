# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:13:40 2017

@author: yashr
"""
import pandas as pd
import numpy as np
import seaborn as sns











#creates a dataframe that only tracks the year something was published and its rating
tempYearData = pd.DataFrame(test_df['details.yearpublished'])
tempYearData = tempYearData.join(test_df['stats.average'])

#drop all data from before 1900 to make the scale on the graph easier to understand
tempYearData.drop(tempYearData[tempYearData['details.yearpublished'] < 1900].index, inplace=True)
tempYearData.drop(tempYearData[tempYearData['details.yearpublished'] > 2017].index, inplace=True)

yearData = tempYearData.groupby('details.yearpublished', as_index=False).mean()


#creates a dataframe that tracks the maximum players and the rating of a board game
maxPlayerData_temp = pd.DataFrame(df['details.maxplayers'])
maxPlayerData_temp = maxPlayerData_temp.join(df['stats.average'])

#Unused code: keep only the ones that are within +3 to -3 standard deviations
"""
notOutlier = np.abs(maxPlayerData['details.maxplayers']-maxPlayerData['details.maxplayers'].mean())<=(3*maxPlayerData['details.maxplayers'].std())
maxPlayerData = maxPlayerData[notOutlier] 
"""
maxPlayerData_temp = maxPlayerData_temp[maxPlayerData_temp['details.maxplayers'] <= 10] 
maxPlayerData_temp = maxPlayerData_temp[maxPlayerData_temp['stats.average'] > 0] 

maxPlayerData = maxPlayerData_temp.groupby('details.maxplayers', as_index=False).mean()

#draw a plot
#g = sns.jointplot(x='details.yearpublished', y='stats.average',dropna=True, data = yearData, alpha=0.01)
yearPlot = sns.jointplot(x='details.yearpublished', y='stats.average',dropna=True, data = yearData)
yearPlot.savefig('plots/YearPlot.png')

playerPlot = sns.jointplot(x='details.maxplayers', y='stats.average',dropna=True, data = maxPlayerData)    
playerPlot.savefig('plots/PlayerPlot.png')
