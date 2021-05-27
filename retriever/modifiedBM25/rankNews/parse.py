__author__ = 'Nick Hirakawa'

import re
import pandas as pd
import json
import argparse

class PickleParser:

	def __init__(self, filename=None, data=None):
		if filename is not None:
			self.df = pd.read_pickle(filename)
		elif data is not None:
			self.df = data
		self.queries = dict()

	def parse(self):
		for index, row in self.df.iterrows():
			if len(row.entity) > 0:
				self.queries[row.id] = row.entity

	def get_queries(self):
		return self.queries

class MatchParser:
	def __init__(self, filename):
		self.filename = filename
		self.entity_match = dict()

	def parse(self):
		self.entity_match = json.load(open(self.filename, "r"))

	def get_map(self):
		return self.entity_match


if __name__ == '__main__':
	qp = MatchParser('C:/Users/xw/Documents/Twitter-analysis/BM25/text/matches2018-10-01.json')
	qp.parse()
	news = qp.get_map()

	print(qp.get_queries())
	print("finish")