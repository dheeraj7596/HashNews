__author__ = 'Nick Hirakawa'

from rankNews.invdx import build_data_structures
from rankNews.rank import score_BM25, new_score_BM25
import collections, functools, operator
import time


class QueryProcessor:
    def __init__(self, queries, corpus, entity_mapping):
        self.queries = queries
        self.entity_mapping = entity_mapping
        self.index, self.dlt = build_data_structures(corpus)

    def run(self):
        results = dict()
        count = 0
        start = time.time()
        for query_id in self.queries:
            results[query_id] = self.run_query(self.queries[query_id])
            count = count + 1
            if count % 10 == 0:
                end = time.time()
                print("processed: %d of %d using %f" % (count, len(self.queries), (end - start)))

        return results

    def get_dict_entities(self, entities):
        doc_dict = [self.index[term] for term in entities if term in self.index]
        result = {}
        try:
            result = dict(functools.reduce(operator.add,
                                           map(collections.Counter, doc_dict)))
        except:
            print(doc_dict)
        return result

    def get_distribution_of_entity(self):
        num_of_doc = [['entity', 'num_of_docs', 'occurence']]
        for entity in self.entity_mapping:
            entities = self.entity_mapping[entity]
            doc_dict = self.get_dict_entities(entities)
            num_of_doc.append([entity, len(doc_dict), sum(doc_dict.values())])
        return num_of_doc

    def run_query(self, query):
        query_result = dict()
        for term in query:
            if term not in self.entity_mapping:
                continue
            entities = self.entity_mapping[term]
            doc_dict = self.get_dict_entities(entities)  # retrieve index entry
            for docid, freq in doc_dict.items():  # for each document and its word frequency
                score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
                                   dl=self.dlt.get_length(docid),
                                   avdl=self.dlt.get_average_length())  # calculate score
                if docid in query_result:  # this document has already been scored once
                    query_result[docid] += score
                else:
                    query_result[docid] = score
        return query_result


class NewQueryProcessor:
    def __init__(self, queries, corpus, randomnews, entity_mapping, random_entity_mapping):
        self.queries = queries
        self.entity_mapping = entity_mapping
        self.randomnews = randomnews
        self.random_entity_mapping = random_entity_mapping
        self.index, self.dlt = build_data_structures(corpus)
        self.randindex, self.randdlt = build_data_structures(randomnews)

    def run(self):
        results = dict()
        count = 0
        start = time.time()
        for query_id in self.queries:
            results[query_id] = self.run_query(self.queries[query_id])
            count = count + 1
            if count % 10 == 0:
                end = time.time()
                print("processed: %d of %d using %f" % (count, len(self.queries), (end - start)))

        return results

    # for each entity, matches to several news entities, get all the docs and num that contain any of the entities
    def get_dict_entities(self, entities):
        doc_dict = [self.index[term] for term in entities if term in self.index]
        result = {}
        try:
            result = dict(functools.reduce(operator.add,
                                           map(collections.Counter, doc_dict)))
        except:
            print(doc_dict)
        return result

    def get_dict_entities_for_randomnews(self, entities):
        doc_dict = [self.randindex[term] for term in entities if term in self.randindex]
        result = set().union(*(d.keys() for d in doc_dict))
        return len(result)

    def get_distribution_of_entity(self):
        num_of_doc = [['entity', 'num_of_docs', 'occurence']]
        for entity in self.entity_mapping:
            entities = self.entity_mapping[entity]
            doc_dict = self.get_dict_entities(entities)
            num_of_doc.append([entity, len(doc_dict), sum(doc_dict.values())])
        return num_of_doc

    def run_query(self, query):
        query_result = dict()
        for term in query:
            if term not in self.entity_mapping:
                continue
            # get the news entities matched to the tweet entity
            entities = self.entity_mapping[term]
            doc_dict = self.get_dict_entities(entities)  # retrieve index entry
            if term not in self.random_entity_mapping:
                rand_n = 1
            else:
                rand_entities = self.random_entity_mapping[term]
                rand_n = self.get_dict_entities_for_randomnews(rand_entities)
            for docid, freq in doc_dict.items():  # for each document and its word frequency
                score = new_score_BM25(n=len(doc_dict), rand_n=rand_n, f=freq, qf=1, r=0, N=len(self.dlt),
                                       rand_N=len(self.randdlt), dl=self.dlt.get_length(docid),
                                       avdl=self.dlt.get_average_length())  # calculate score
                if docid in query_result:  # this document has already been scored once
                    query_result[docid] += score
                else:
                    query_result[docid] = score
        return query_result
