import pandas as pd
import numpy as np
import re
import string
import joblib
import datetime as dt
import html
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


# Preprocess & Cleaning
# Function to remove Stopwords
def remove_stopwords(row):
    stopword_list = stopwords.words('english')
    words = []
    for word in row:
        if word not in stopword_list:
            words.append(word)
    return words


# Function to remove emojis
def remove_emoji(tweets):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweets)


# Preprocessing Function
def tweets_preprocessing(raw_df):

    # Removing all tickers from comments
    raw_df['Message'] = raw_df['Message'].str.replace(r'([$][a-zA-z]{1,5})', '')

    # Make all sentences small letters
    raw_df['Message'] = raw_df['Message'].str.lower()

    # Converting HTML to UTF-8
    raw_df["Message"] = raw_df["Message"].apply(html.unescape)

    # Removing hastags, mentions, pagebreaks, handles
    # Keeping the words behind hashtags as they may provide useful information about the comments e.g. #Bullish #Lambo
    raw_df["Message"] = raw_df["Message"].str.replace(r'(@[^\s]+|[#]|[$])', ' ')  # Replace '@', '$' and '#...'
    raw_df["Message"] = raw_df["Message"].str.replace(r'(\n|\r)', ' ')  # Replace page breaks

    # Removing https, www., any links etc
    raw_df["Message"] = raw_df["Message"].str.replace(r'((https:|http:)[^\s]+|(www\.)[^\s]+)', ' ')

    # Removing all numbers
    raw_df["Message"] = raw_df["Message"].str.replace(r'[\d]', '')

    # Remove emoji
    raw_df["Message"] = raw_df["Message"].apply(lambda row: remove_emoji(row))

    # Tokenization
    raw_df['Message'] = raw_df['Message'].apply(word_tokenize)

    # Remove Stopwords
    raw_df['Message'] = raw_df['Message'].apply(remove_stopwords)

    # Remove Punctuation
    raw_df['Message'] = raw_df['Message'].apply(lambda row: [word for word in row if word not in string.punctuation])

    # Combining back to full sentences
    raw_df['Message'] = raw_df['Message'].apply(lambda row: ' '.join(row))

    # Remove special punctuation not in string.punctuation
    raw_df['Message'] = raw_df['Message'].str.replace(r"\“|\”|\‘|\’|\.\.\.|\/\/|\.\.|\.|\"|\'", '')

    # Remove all empty rows
    processed_df = raw_df[raw_df['Message'].str.contains(r'^\s*$') == False]

    return processed_df


# Classification
# Input processed dataframe into model
# Shows classification report and confusion matrix
def classification_report(processed_df, model=joblib.load("stocktwits_modelNB.pkl")):

    # Getting Precision, Accuracy score from model trained on SPY comments for TSLA
    test_data = processed_df[processed_df["Sentiment"].isin(["Bearish", "Bullish"])]
    test_preds = model.predict(test_data['Message'])
    accuracy_score(test_data['Sentiment'], test_preds)

    print('accuracy score: ', accuracy_score(test_data['Sentiment'], test_preds))
    print('\n')
    print('confusion matrix: \n', confusion_matrix(test_data['Sentiment'], test_preds))
    print('\n')
    print(classification_report(test_data['Sentiment'], test_preds))


# Using model to classify processed Tweets
def classify_tweets(processed_df, model):
    processed_df["ML Sentiment"] = model.predict(processed_df['Message'])

    # Combining ST labelled sentiment & ML Labelled sentiment
    processed_df["Combined Sentiment"] = " "
    processed_df.loc[processed_df["Sentiment"] == "N/A", "Combined Sentiment"] = processed_df['ML Sentiment']
    processed_df.loc[processed_df["Sentiment"].isin(["Bullish", "Bearish"]),
                     "Combined Sentiment"] = processed_df['Sentiment']

    classified_df = processed_df[["User_id", "Message", "Date", "Time", "Combined Sentiment"]]
    return classified_df


# Narrowing down into pre-market slots and trading days
# Function to get nearest Monday for Fri/Sat/Sun
def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return d + dt.timedelta(days_ahead)


# Function to filter post-classification dataframe into trading days and pre-market tweets
def filtering_trading_days(classified_df):
    public_hol = ["2020-01-01", "2020-01-20", "2020-02-17", "2020-04-10",
                  "2020-05-25", "2020-07-03", "2020-09-07", "2020-11-26",
                  "2020-12-25"]

    # Cleaning of Date Time
    classified_df["Time"] = classified_df["Time"].apply(lambda x: x.split("Z")[0])
    classified_df["Hour of Day"] = pd.to_datetime(classified_df["Time"], format='%H:%M:%S').dt.hour.astype(str)
    classified_df.loc[classified_df["Hour of Day"].str.len() == 1, "Hour of Day"] = "0" + classified_df["Hour of Day"]
    classified_df["Date_Month"] = pd.to_datetime(classified_df["Date"]).dt.day
    classified_df["Date_Month"] = pd.to_datetime(classified_df["Date"]).dt.month_name() + "-" + \
                                  classified_df["Date_Month"].astype(str)
    classified_df["Date_Time"] = classified_df["Date_Month"] + " / " + classified_df["Hour of Day"].astype(str)
    classified_df['Date'] = pd.to_datetime(classified_df['Date'])
    classified_df["Day_of_week"] = pd.to_datetime(classified_df["Date"]).dt.day_name()
    classified_df["Time"] = pd.to_datetime(classified_df["Time"], format='%H:%M:%S')

    # FILTER CONDITIONS
    # Mon - thurs after 9pm -> becomes next day pre market data
    c1 = (classified_df['Day_of_week'].isin(["Monday", "Tuesday", "Wednesday", "Thursday"])) & \
         (classified_df["Time"].dt.strftime('%H:%M:%S') > '21:00:00')

    r1 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=1)).dt.date

    # Friday post 9pm, Saturday and sunday -> Becomes aggregated to Monday pre market data
    c2 = (classified_df['Day_of_week'].isin(["Sunday", "Saturday"])) | \
         ((classified_df['Day_of_week'].isin(["Friday"])) &
          (classified_df["Time"].dt.strftime('%H:%M:%S') > '21:00:00'))

    r2 = classified_df["Date"].apply(lambda x: next_weekday(x, 0)).astype(str)

    # Today is mon,tues,wed // tomorrow is a holiday // get Afterhour data after market close today 
    c3 = classified_df['Day_of_week'].isin(["Monday", "Tuesday", "Wednesday"]) & \
         (classified_df["Date"] + pd.Timedelta(days=1)).isin(public_hol) & \
         (classified_df["Time"].dt.strftime('%H:%M:%S') > '21:00:00')

    r3 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=2)).dt.date

    # Today is thurs // tomorrow is a holiday // get Afterhour data after market close today 
    c4 = classified_df['Day_of_week'].isin(["Thursday"]) & \
         (classified_df["Date"] + pd.Timedelta(days=1)).isin(public_hol) & \
         (classified_df["Time"].dt.strftime('%H:%M:%S') > '21:00:00')

    r4 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=4)).dt.date

    # Today is friday // next monday is a holiday // get Afterhour data after market close today 
    c5 = classified_df['Day_of_week'].isin(["Friday"]) & \
         (classified_df["Date"] + pd.Timedelta(days=3)).isin(public_hol) & \
         (classified_df["Time"].dt.strftime('%H:%M:%S') > '21:00:00')

    r5 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=4)).dt.date

    # Today is saturday // next monday is a holiday 
    c6 = classified_df['Day_of_week'].isin(["Saturday"]) & \
         (classified_df["Date"] + pd.Timedelta(days=2)).isin(public_hol)

    r6 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=3)).dt.date

    # Today is sunday // next monday is a holiday
    c7 = classified_df['Day_of_week'].isin(["Sunday"]) & (classified_df["Date"] + pd.Timedelta(days=1)).isin(public_hol)

    r7 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=2)).dt.date

    # Today is mon,tues,wed,thurs // today is a holiday
    c8 = classified_df["Date"].isin(public_hol) & \
         classified_df['Day_of_week'].isin(["Monday", "Tuesday", "Wednesday", "Thursday"])

    r8 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=1)).dt.date

    # Today is friday // # today is a holiday
    c9 = classified_df["Date"].isin(public_hol) & classified_df['Day_of_week'].isin(["Friday"])

    r9 = pd.to_datetime(classified_df["Date"] + pd.Timedelta(days=3)).dt.date

    # Create Column to distinguish comments to trading days
    classified_df["Pre-Market Date"] = np.select([c9, c8, c7, c6, c5, c4, c3, c2, c1],
                                                 [r9, r8, r7, r6, r5, r4, r3, r2, r1],
                                                 default=classified_df['Date'].astype(str))
    classified_df["PM_Date_Month"] = pd.to_datetime(classified_df["Pre-Market Date"]).dt.day
    classified_df["PM_Date_Month"] = pd.to_datetime(classified_df["Pre-Market Date"]).dt.month_name() + \
                                     "-" + classified_df["PM_Date_Month"].astype(str)
    classified_df = classified_df.loc[(classified_df["PM_Date_Month"] != "January-1") &
                                      (classified_df["Date"] <= "2020-12-31")]

    # Filter out outside market hours (Before 2.30pm , after 9pm UTC+0)
    trade_hrs_df = classified_df[classified_df["Time"].dt.strftime('%H:%M:%S').between('00:00:00', '14:29:59') |
                                 classified_df["Time"].dt.strftime('%H:%M:%S').between('21:00:00', '23:59:59')]

    return trade_hrs_df


# Function to derive Bull-bear ratio
def bull_bear_ratio(trade_hrs_df):
    bull_bear_df = trade_hrs_df.groupby(
        ["PM_Date_Month", 'Combined Sentiment']).agg({"Message": "count"}).unstack().reset_index()
    bull_bear_df['PM_Date_Month'] = bull_bear_df['PM_Date_Month'].apply(lambda date: datetime.strptime(date, '%B-%d'))
    bull_bear_df.sort_values("PM_Date_Month", inplace=True)
    bull_bear_df['PM_Date_Month'] = bull_bear_df['PM_Date_Month'].dt.strftime('%B-%d')
    bull_bear_df.set_index('PM_Date_Month', inplace=True)
    bull_bear_df['Bull/Bear Ratio'] = (bull_bear_df[('Message', 'Bullish')]) / (bull_bear_df[('Message', 'Bearish')])

    return bull_bear_df


def merge_price_sentiment(price_df, bull_bear_df, ema_list=None):

    # Merge daily score with SPY price
    if ema_list is None:
        ema_list = [5, 6, 7, 8, 9, 10, 15, 20]
    combined_df = bull_bear_df.merge(price_df, how="left", left_index=True, right_index=True)

    combined_df.columns = ["Bearish", "Bullish", "Bull/Bear Ratio",
                "Date", "High", "Low", "Open",
                "Close", "Volume", "Adjusted Close",
                "PM_change", "Day_change", "%_Change"]

    for ema in ema_list:
        exp_ema = combined_df['Bull/Bear Ratio'].ewm(span=ema, min_periods=ema, adjust=False).mean()
        combined_df[f"Bull/Bear Ratio EMA {ema}"] = exp_ema

    combined_df["% of Bullish"] = round((combined_df['Bullish'] * 100) /
                                        (combined_df['Bullish'] + combined_df['Bearish']), 2)
    combined_df["% of Bearish"] = round((combined_df['Bearish'] * 100) /
                                        (combined_df['Bullish'] + combined_df['Bearish']), 2)
    combined_df['Middle line'] = 50
    return combined_df


# In[200]:


def backtest_results(combined_df, ema_list=None):

    if ema_list is None:
        ema_list = [5, 6, 7, 8, 9, 10, 15, 20]

    overall_dict = {}  # Dictionary for all the dfs

    for ema in ema_list:
        net_cash = 10000
        start_port_val = net_cash
        net_shares = 0
        net_profit = 0
        interim_master_list = []

        for row in range(len(combined_df)):

            bb_ratio = combined_df["Bull/Bear Ratio"][row]
            bbratio_ema = combined_df[f"Bull/Bear Ratio EMA {ema}"][row]

            # Buy Signal
            if (bb_ratio > bbratio_ema) & (net_shares <= 0):
                trx_dict = {}
                date = combined_df["Date"][row]
                open_px = combined_df["Open"][row]
                close_px = combined_df["Adjusted Close"][row]

                # To prevent miscalculation for day 0
                if net_shares == 0:
                    shares_bought = int(net_cash / open_px)  # Rounded down
                    pre_market_profit = 0
                else:
                    shares_bought = int(net_cash / open_px)
                    pre_market_profit = net_shares * combined_df["PM_change"][row]

                net_shares = net_shares + shares_bought
                net_cash = net_cash % open_px
                profit = net_shares * combined_df["Day_change"][row]
                day_profit = pre_market_profit + profit
                net_profit += day_profit
                portfolio_val = start_port_val + net_profit

                trx_dict[f"Date EMA {ema}"] = date
                trx_dict[f"Portfolio Value EMA {ema}"] = portfolio_val
                trx_dict[f"Net Shares EMA {ema}"] = net_shares
                trx_dict[f"Net Cash EMA {ema}"] = net_cash
                trx_dict[f"Action EMA {ema}"] = f"BUY {shares_bought} at Open"
                trx_dict[f"PM Profit EMA {ema}"] = pre_market_profit
                trx_dict[f"Trade Session Profit EMA {ema}"] = profit
                trx_dict[f"Total Day Profit EMA {ema}"] = day_profit
                trx_dict[f"Current Net Profit EMA {ema}"] = net_profit
                trx_dict[f"Adjusted Close EMA {ema}"] = close_px
                interim_master_list.append(trx_dict)

            # Short signal
            elif (bb_ratio < bbratio_ema) & (net_shares >= 0):
                trx_dict = {}
                date = combined_df["Date"][row]
                open_px = combined_df["Open"][row]
                close_px = combined_df["Adjusted Close"][row]

                if net_shares == 0:
                    shares_sold = int(net_cash / open_px)
                    pre_market_profit = 0

                else:
                    shares_sold = (2 * net_shares)
                    pre_market_profit = net_shares * combined_df["PM_change"][row]

                net_shares = net_shares - shares_sold
                net_cash = net_cash + (shares_sold * open_px)
                profit = net_shares * combined_df["Day_change"][row]
                day_profit = pre_market_profit + profit
                net_profit += day_profit
                portfolio_val = start_port_val + net_profit

                trx_dict[f"Date EMA {ema}"] = date
                trx_dict[f"Portfolio Value EMA {ema}"] = portfolio_val
                trx_dict[f"Net Shares EMA {ema}"] = net_shares
                trx_dict[f"Net Cash EMA {ema}"] = net_cash
                trx_dict[f"Action EMA {ema}"] = f"SELL {shares_sold} at Open"
                trx_dict[f"PM Profit EMA {ema}"] = pre_market_profit
                trx_dict[f"Trade Session Profit EMA {ema}"] = profit
                trx_dict[f"Total Day Profit EMA {ema}"] = day_profit
                trx_dict[f"Current Net Profit EMA {ema}"] = net_profit
                trx_dict[f"Adjusted Close EMA {ema}"] = close_px
                interim_master_list.append(trx_dict)

            else:
                trx_dict = {}
                date = combined_df["Date"][row]
                close_px = combined_df["Adjusted Close"][row]

                # Prevent NaN for day 0
                if net_shares == 0:
                    pre_market_profit = 0

                else:
                    pre_market_profit = net_shares * combined_df["PM_change"][row]

                net_cash = net_cash
                profit = net_shares * combined_df["Day_change"][row]
                day_profit = pre_market_profit + profit
                net_profit += day_profit
                portfolio_val = start_port_val + net_profit

                trx_dict[f"Date EMA {ema}"] = date
                trx_dict[f"Portfolio Value EMA {ema}"] = portfolio_val
                trx_dict[f"Net Shares EMA {ema}"] = net_shares
                trx_dict[f"Net Cash EMA {ema}"] = net_cash
                trx_dict[f"Action EMA {ema}"] = f"No Trade"
                trx_dict[f"PM Profit EMA {ema}"] = pre_market_profit
                trx_dict[f"Trade Session Profit EMA {ema}"] = profit
                trx_dict[f"Total Day Profit EMA {ema}"] = day_profit
                trx_dict[f"Current Net Profit EMA {ema}"] = net_profit
                trx_dict[f"Adjusted Close EMA {ema}"] = close_px
                interim_master_list.append(trx_dict)

        overall_dict[f'aapl_strat_EMA_{ema}'] = pd.DataFrame(interim_master_list)

    aapl_strat = pd.concat([overall_dict['aapl_strat_EMA_5'], overall_dict['aapl_strat_EMA_6'],
                            overall_dict['aapl_strat_EMA_7'], overall_dict['aapl_strat_EMA_8'],
                            overall_dict['aapl_strat_EMA_9'], overall_dict['aapl_strat_EMA_10'],
                            overall_dict['aapl_strat_EMA_15'], overall_dict['aapl_strat_EMA_20']], axis=1)

    return aapl_strat

# Import Tweets & Price
df = pd.read_pickle("ST_AAPL_raw.pkl")
aapl_price_df = pd.read_pickle("AAPL_Daily_yf.pkl")
trained_model = joblib.load("stocktwits_modelNB.pkl")

if __name__ == "__main__":
    cleaned_df = tweets_preprocessing(df)  # Clean
    print('Cleaned')
    sentiment_df = classify_tweets(cleaned_df, trained_model)  # Classify
    print('Classified')
    filtered_df = filtering_trading_days(sentiment_df)  # Filter
    print('Filtered')
    bb_df = bull_bear_ratio(filtered_df)  # Evaluate
    print('Grouped')
    merge_df = merge_price_sentiment(aapl_price_df, bb_df)  # Combine
    print('Merged')
    aapl_results = backtest_results(merge_df)  # Backtest
    print('Backtested')
    print(aapl_results)




