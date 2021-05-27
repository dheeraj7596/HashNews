# this is for topic model
from gensim import corpora
import pickle
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()


def tokenize(text):
    """this function is to tokenize the headline into a list of individual words"""
    return text.split(" ")


def get_lemma(word):
    """this function is to lemmatize the words in a headline into its root form"""
    lemma = lemmatizer.lemmatize(word)
    if lemma is None:
        return word
    else:
        return lemma


def prepare_text_for_lda(text):
    tokens = tokenize(text)  # parse and tokenize the headline into a list of words
    tokens = [get_lemma(token) for token in tokens]  # lemmatize the words in the headline
    return tokens


def read_file(file_path):
    f = open(file_path, "r", encoding='utf-8')
    lines = f.readlines()
    f.close()
    return [l.rstrip("\n") for l in lines]




if __name__ == '__main__':
    key = "test"
    base_path = "/data1/xiuwen/twitter/tweet2020/tweet-without-conversation/"
    raw_data_path = base_path + key + "_post.txt"
    raw_data = read_file(raw_data_path)
    text_data = [prepare_text_for_lda(i) for i in raw_data]
    dictionary = corpora.Dictionary(text_data)  # Convert all headlines into a corpus of words, with each word as a token
    corpus = [dictionary.doc2bow(text) for text in text_data]  # Convert each headline (a list of words) into the bag-of-words format. (Word ID, Count of word)
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word=dictionary, passes=15, minimum_probability=0.0)
    result = []
    for i in range(len(corpus)):
        result.append(ldamodel[corpus[i]])
    pickle.dump(result, open(base_path + "topic_model_" + key + ".pkl", "wb"))
