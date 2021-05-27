import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')


from rankNews.query import NewQueryProcessor
import operator
import pandas as pd
import argparse
from rankNews.parse import MatchParser, PickleParser
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='pass day')
parser.add_argument('--day', '-d')
args = parser.parse_args()

# def get_news_5days_before_tweet(day):
#     is_date = tweets.created_at == day
#     tweets_this_day = tweets[is_date]
#     tweet_index = tweets_this_day.id.to_list()
#     news_within_range = news[(news.publishdate <= day) & (news.publishdate >= (day - timedelta(days=5)))]
#     return tweet_index, news_within_range.index.to_list()


# def get_news_before_tweet(news, date_range):
#     news_within_range = news[(news.publishdate < day) & (news.publishdate >= (day - timedelta(days=date_range)))]
#     return news_within_range

def get_tweet_and_news_5days(tweets, news, day):
	is_date = tweets.created_at == day
	tweets_this_day = tweets[is_date]
	# tweet_index = tweets_this_day.id.to_list()
	news_within_range = news[(news.publishdate <= day) & (news.publishdate >= (day - timedelta(days=5)))]
	return tweets_this_day[['id', 'entity']], news_within_range[['id', 'entity']]


def main():
	if int(args.day) > 24:
		day = datetime(2020, 5, int(args.day)).date()
	else:
		day = datetime(2020, 6, int(args.day)).date()

	# day = datetime(2018, 10, int(args.day)).date()
	data_path = "/data1/xiuwen/twitter/"

	news = pd.read_pickle(data_path + "tweet2020/news.pkl")
	tweets = pd.read_pickle(data_path + "tweet2020/tweets.pkl")
	tweets['created_at'] = pd.to_datetime(tweets.created_at, errors='coerce').dt.date
	news['publishdate'] = pd.to_datetime(news.publishdate, errors='coerce').dt.date
	random_news_path = "news_random.pkl"
	result_path = "/home/xiuwen/tweetAnalyze/result2020/"

	entity_path = "entity_matches.json"
	ran_entity_path = "entity_matches_random.json"

	this_tweet, this_news = get_tweet_and_news_5days(tweets, news, day)


	qp = PickleParser(data=this_tweet)
	cp = PickleParser(data=this_news)
	rp = PickleParser(filename=data_path+random_news_path)

	entity_mapping = MatchParser(result_path+entity_path)
	rand_entity_mapping = MatchParser(result_path+ran_entity_path)

	# get queries
	qp.parse()
	queries = qp.get_queries()
	print(len(queries))
	# get news
	cp.parse()
	corpus = cp.get_queries()
	print(len(corpus))
	# get random news
	rp.parse()
	randomnews = rp.get_queries()
	print(len(randomnews))
	# get tweet-news entity match
	entity_mapping.parse()
	# get tweet-randomnews entity match
	rand_entity_mapping.parse()

	proc = NewQueryProcessor(queries, corpus, randomnews, entity_mapping.get_map(), rand_entity_mapping.get_map())
	results = proc.run()
	dataset = [['tweet_id', 'news_id', 'score']]
	for qid in results:
		result = results[qid]
		sorted_x = sorted(result.items(), key=operator.itemgetter(1))
		# sorted_x.reverse()
		if result is None or len(result) == 0:
			print("no match")
			continue
		news_id = sorted_x[-1][0]
		score = sorted_x[-1][1]
		# news_id = [i[0] for i in sorted_x[:5]]
		# score = [i[1] for i in sorted_x[:5]]
		dataset.append([qid, news_id, score])
	df_result = pd.DataFrame(dataset[1:], columns=dataset[0])
	df_result.to_pickle(result_path + "modified_news_entity_match" + str(day) + ".pkl")


if __name__ == '__main__':
	main()
