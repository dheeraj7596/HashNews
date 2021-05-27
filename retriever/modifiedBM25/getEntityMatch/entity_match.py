# get entity_matches.json and entity_matches_random.json
import pandas as pd
from collections import defaultdict
import json
import argparse
import time
from datetime import datetime, timedelta

# entity_path = "../data/"
# tweet_path = "ne_tweets_ner.txt"
# news_path = "ne_randomnews.txt"
data_path = "/data1/xiuwen/twitter/"
result_path = "/home/xiuwen/tweetAnalyze/"
# result_path = common_result_path

# news_csv = "randomnews.csv"
# tweet_json = "tweet.json"


def extract_entity_set(df):
    ner_set = set()
    for en in df.entity.to_list():
        ner_set.update(en)
    return ner_set


def extract_entity_set_from_json(df, key):
    ner = [['id', 'entity']]
    ner_set = set()
    for index, row in df.iterrows():
        temp = [i['ner'].lower() for i in row[key]]
        ner_set.update(temp)
        ner.append([row['id'] , temp])
    return pd.DataFrame(ner[1:], columns=ner[0]), ner_set


def entity_match(t_entity, n_entity):
    matches = defaultdict(list)
    for t in t_entity:
        t_temp = t.split()
        for n in n_entity:
            n_temp = n.split()
            if len([1 for i in t_temp if i in n_temp])/len(t_temp) >= 0.65:
                matches[t].append(n)
    return matches


# parser = argparse.ArgumentParser(description='pass day')
# parser.add_argument('--day', '-d')
# args = parser.parse_args()


if __name__ == '__main__':
    # use this for first dataset
    # day = datetime(2018, 10, int(args.day)).date()
    # if (int(args.day) > 24):
    #     day = datetime(2020, 5, int(args.day)).date()
    # else:
    #     day = datetime(2020, 6, int(args.day)).date()
    # print(str(day))
    # rand_news = pd.read_pickle(data_path+"news_random.pkl")
    # news = pd.read_pickle(data_path + "tweet_2020/news.pkl")
    year = "2020"
    news = pd.read_json(data_path+"tweet"+year+"/ne_news.txt", orient="records", lines=True)
    # rand_news = pd.read_json(data_path + "ne_news.txt", orient="records", lines=True)
    rand_news = pd.read_pickle(data_path + "news_random.pkl")
    tweet = pd.read_pickle(data_path + "tweet" + year + "/tweets.pkl")

    # news = pd.read_csv(data_path+news_csv, encoding="utf-8", error_bad_lines=False)
    # news['publishdate'] = pd.to_datetime(news.publishdate, errors='coerce').dt.date
    # news = news.dropna()
    # tweets = pd.read_json(data_path + tweet_json)
    # tweets.created_at = tweets.created_at.dt.date
    # tweets_entity_data = pd.read_json(entity_path + tweet_path, orient="records", lines=True)
    # news_entity_data = pd.read_json(entity_path + news_path, orient="records", lines=True)
    #
    # tweets_index, news_index = get_news_5days_before_tweet(day)
    # is_date = tweets.created_at == day
    # tweets_this_day = tweets[is_date]
    # tweets_index = tweets_this_day.id.to_list()

    tweet_entity = extract_entity_set(tweet)
    news_pkl, news_entity = extract_entity_set_from_json(news, 'news')
    rand_news_entity = extract_entity_set(rand_news)
    # rand_pkl, rand_news_entity = extract_entity_set_from_json(rand_news, 'news')
    news_pkl.to_pickle(data_path+"tweet"+year+"/news_entity.pkl")
    # rand_news.to_pickle(data_path + "news_random.pkl")
    # print(len(quries))
    # print(len(corpus))
    print(len(tweet_entity))
    print(len(news_entity))
    print(len(rand_news_entity))

    # match = 1
    # mismatch = -1
    # scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    # test = list(tweet_entity)[:20]
    # sw = swalign.LocalAlignment(scoring)
    matches = entity_match(tweet_entity, news_entity)
    random_matches = entity_match(tweet_entity, rand_news_entity)
    match_json = json.dumps(matches)
    rand_match_json = json.dumps(random_matches)
    # print(json)
    # quries.to_pickle("./result/tweets.pkl")
    # corpus.to_pickle("./result/news.pkl")
    f = open(result_path+"result"+year+"/entity_matches.json", "w")
    f.write(match_json)
    f.close()
    f = open(result_path + "result"+year+"/entity_matches_random.json", "w")
    f.write(rand_match_json)
    f.close()

