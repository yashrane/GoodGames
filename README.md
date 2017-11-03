# GoodGames
Uses Python's Pandas library to predict the rating of a Board game using a dataset of board game statistics



### GoodGames.py
* Reads the sqlite file into a dataframe and cleans the data
* Uses a Random Forest to predict the rating of boardgames 
* In the future, AdaBoost or a boosted gradient descent algorithm may be used for predictions instead


### MakePlots.py 
* Generates plots using the characteristics the Random Forest from GoodGames.py identified as important

#### Year Published vs Average Rating
![Year Plot](https://raw.githubusercontent.com/yashrane/GoodGames/master/lib/YearPlot.png "YearPlot")

The most obvious thing about this plot at first glance is the difference in how the data is distributed. The earlier the year published, the more spread out the data is. This may be due to bias in the data, but without another dataset to check this against, I can't verify that.
There seems to be a general trend for the ratings of games to increase starting in the year 2000. It grows extremely quickly, with the average rating jumping from about 5.5 to nearly 8 from 2000 to 2017. From this, I can conclude that people tend to prefer newer games over older ones.



#### Max Players vs Average Rating
![Player Plot](https://raw.githubusercontent.com/yashrane/GoodGames/master/lib/PlayerPlot.png "PlayerPlot")

There are two spikes in the rating here: one at 1 player and another at 5 players. These seem to be the optimal maximum number of players, as the rating drops off linearly after each of these spikes. 6 and 8 players are especially notable, as they are drastically lower than the surrounding points.