import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
from rankNews.query import QueryProcessor
import operator
import pandas as pd
import argparse
from rankNews.parse import MatchParser, PickleParser
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='pass day')
parser.add_argument('--day', '-d')
args = parser.parse_args()


def get_tweet_and_news_5days(tweets, news, day):
    is_date = tweets.created_at == day
    tweets_this_day = tweets[is_date]
    # tweet_index = tweets_this_day.id.to_list()
    news_within_range = news[(news.publishdate <= day) & (news.publishdate >= (day - timedelta(days=5)))]
    return tweets_this_day[['id', 'entity']], news_within_range[['id', 'entity']]


def main():
    # day = datetime(2018, 10, int(args.day)).date()
    if int(args.day) > 24:
        day = datetime(2020, 5, int(args.day)).date()
    else:
        day = datetime(2020, 6, int(args.day)).date()
    print(str(day))
    data_path = "/data1/xiuwen/twitter/tweet2020/"
    result_path = "/home/xiuwen/tweetAnalyze/BM25/result2020/"
    # result_path = common_result_path + "withouttime/"

    news = pd.read_pickle(data_path + "news.pkl")
    tweets = pd.read_pickle(data_path + "tweets.pkl")
    tweets['created_at'] = pd.to_datetime(tweets.created_at, errors='coerce').dt.date
    entity_path = "entity_matches.json"

    this_tweet, this_news = get_tweet_and_news_5days(tweets, news, day)

    print(len(this_news))
    print(len(this_tweet))
    qp = PickleParser(data=this_tweet)
    cp = PickleParser(data=this_news)
    entity_mapping = MatchParser(result_path + entity_path)

    qp.parse()
    queries = qp.get_queries()
    cp.parse()
    print(len(queries))
    corpus = cp.get_queries()
    entity_mapping.parse()
    proc = QueryProcessor(queries, corpus, entity_mapping.get_map())
    results = proc.run()
    dataset = [['tweet_id', 'news_id', 'score']]
    for qid in results:
        result = results[qid]
        sorted_x = sorted(result.items(), key=operator.itemgetter(1))
        sorted_x.reverse()
        news_id = [i[0] for i in sorted_x[:5]]
        score = [i[1] for i in sorted_x[:5]]
        dataset.append([qid, news_id, score])
    df_result = pd.DataFrame(dataset[1:], columns=dataset[0])
    df_result.to_pickle(result_path + "bm25/news_entity_match" + str(day) + ".pkl")


if __name__ == '__main__':
    main()
