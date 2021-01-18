import requests
import time
import datetime
import pandas as pd
import sys


def scraper(ticker, latest_xhr_id='<INSERT API ID>', max_volume=50000, start_date='2020-01-01', end_date='2020-12-31'):

    # Push Errors if scrape volume less than 0:
    if max_volume <= 0:
        sys.exit("Error: max_volume must be more than 0")

    start = time.time()
    master_content = []  # List to store all data extracted
    scroll_list = [latest_xhr_id]  # List to store all XHR id to be part of the url parameters
    tracker_list = []  # List containing integers for tracking progress
    tracker = 0
    fail_count = 0

    for x in range(5001):
        if x > 0:
            addition = x * 100
            tracker_list.append(addition)

    # Running for loop for collecting data from stocktwits. Each loop collects 20 comments.
    for _ in range(max_volume):
        try:
            headers = {
                <INSERT OWN HEADERS>
            }

            params = (
                <INSERT OWN PARAMS>
            )

            response = requests.get(f'<API LINK>',
                                    headers=headers, params=params)
            content = response.json()
            messages = content['messages']

            # Creating dictionary for items scraped
            for item in messages:
                content_dict = {}
                content_dict['User_id'] = item['id']
                content_dict['Message'] = item['body']
                content_dict['Date'] = item['created_at'].split('T')[0]
                content_dict['Time'] = item['created_at'].split('T')[1]
                try:
                    content_dict['Sentiment'] = item['entities']['sentiment']['basic']
                except TypeError:
                    content_dict['Sentiment'] = "N/A"
                master_content.append(content_dict)

            next_20_id = str(messages[-1]['id'])
            scroll_list.append(next_20_id)

            # Progress Tracker
            tracker += 1

            # Variables for tracker
            last_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            first_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')

            diff = last_date - first_date

            three_quarter_done = first_date + diff/4
            half_done = first_date + diff/2
            one_quarter_done = first_date + diff*3/4

            # Trackers
            for number in tracker_list:
                if tracker == number:
                    print(f"Extracted {number}...")
                    print(f"run time = {time.time() - start}")  # Check run time

            if (master_content[-1]['Time'].split(":")[0] == "00" and
                    master_content[-1]['Date'] == f'{one_quarter_done}'):
                print("25% done")

            elif (master_content[-1]['Time'].split(":")[0] == "00" and
                    master_content[-1]['Date'] == f'{half_done}'):
                print("50% done")

            elif (master_content[-1]['Time'].split(":")[0] == "00" and
                    master_content[-1]['Date'] == f'{three_quarter_done}'):
                print("75% done")

            elif (master_content[-1]['Time'].split(":")[0] == "00" and
                    master_content[-1]['Date'] == f'{first_date}'):
                print("100% done")
                print(f'Number of tweets unable to scrape: {fail_count * 20}')
                break

        except:
            fail_count += 1

    print(f"Number of tweets scraped: {len(master_content)}")
    print(f"Last Tweet: {master_content[-1]}")

    df = pd.DataFrame(master_content)
    return df


if __name__ == "__main__":
    tweets_df = scraper('aapl', max_volume=2)
    print(tweets_df)
