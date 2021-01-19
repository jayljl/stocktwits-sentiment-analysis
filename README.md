# stocktwits-sentiment-analysis
This project aims to scrape tweets data from stocktwits and rely on them to form pre-market sentiments on certain tickers. 
Around 50% of tweets are tagged with sentiments, while 50% are untagged. Hence, supervised learning could be done using those tagged tweets to train a classifier model to tag the remaining untagged tweets.
Aggregating all the sentiments during the pre-market we can form a daily view (Bull/bear) of the retail sentiments. These pre-market sentiments will act as long/short signals for the respective stocks. You may the find the backtested results in the dashboard link below:

The dashboard built using Plotly Dash: https://stocktwits-sentiment-dashboard.herokuapp.com/                              
Medium article explaining the project: https://jayljl.medium.com/mining-stocktwits-retail-sentiments-for-momentum-trading-4594a91833b4 
