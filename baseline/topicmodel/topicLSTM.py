import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Embedding, Dropout, LSTM, AveragePooling1D, Lambda, Concatenate, \
    Multiply, RepeatVector, Flatten, Activation, Permute, merge
import keras.backend as K
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle as pkl
from topicAtt import topicAttention
from preprocess import read_file
import numpy as np
import os

def load_word_vector(path_to_glove_file):
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


def build_matrix():
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix


def myLossFunc(y_true, y_pred):
    probs_log = -K.log(y_pred)
    loss = K.mean(K.sum(probs_log * y_true, axis=-1))
    # loss = K.mean(K.sum(K.clip(probs_log * y_true, -1e40, 100), axis=-1))
    return loss


def modelDef():
    input_text = Input(shape=(seq_length,))
    input_topic = Input(shape=(topic_num,))
    embeddings = Embedding(
        num_words,
        embedding_size,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )(input_text)
    # embeddings = Embedding(input_dim=num_words+index_from, output_dim=embedding_size,
    #                        mask_zero=True, input_length=seq_length)(input_text)
    tFeature = LSTM(units=500, return_sequences=True)(embeddings)
    topic_h = topicAttention()([tFeature, input_topic])
    dropout = Dropout(drop_rate)(topic_h)
    Softmax = Dense(num_tags, activation="softmax", use_bias=True)(dropout)
    model = Model(inputs=[input_text, input_topic], outputs=[Softmax])
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss=myLossFunc)
    return model


def process_labels():
    base_path = "/data1/xiuwen/twitter/tweet2020/tweet-without-conversation/"
    train_tag_path = base_path + "train_repeat_tag.txt"
    train_tag = read_file(train_tag_path)
    train_tag_processed = [''.join(i.split(' ')) for i in train_tag]
    tag_set = set(train_tag_processed)
    num = len(tag_set)
    tag_set_list = list(tag_set)
    index = range(len(tag_set_list))
    index_tag_mapping = dict(zip(index, tag_set_list))
    tag_index_mapping = dict(zip(tag_set_list, index))
    encoding_of_tags = np.zeros((len(train_tag), num))
    for i in range(len(train_tag_processed)):
        encoding_of_tags[i, tag_index_mapping[train_tag_processed[i]]] = 1
    return encoding_of_tags, index_tag_mapping


def write_to_file(file_path, entities):
    f = open(file_path, "w")
    for t in entities:
        f.write(t)
        f.write("\n")
    f.close()


def translate(y_pred, index_tag_mapping):
    base_path = "/data1/xiuwen/twitter/tweet2020/tweet-without-conversation/"
    prediction = []
    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[::-1][:20]
        prediction.append(';'.join([index_tag_mapping[k] for k in top_indices]))
    write_to_file(base_path + "prediction.txt", prediction)
    # return prediction
    # pkl.dump(prediction, open(base_path + "prediction.pkl", "wb"))


def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []
    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:]
        if np.sum(y_true[i, top_indices]) >= 1:
            acc_count += 1
        p = np.sum(y_true[i, top_indices]) / top_K
        r = np.sum(y_true[i, top_indices]) / np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)
    acc_K = acc_count * 1.0 / y_pred.shape[0]
    return acc_K, np.mean(np.array(precision_K)), np.mean(np.array(recall_K)), np.mean(np.array(f1_K))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    seq_length = 40
    batch_size = 100
    embedding_size = 200
    attention_size = 200
    topic_num = 100
    dim_k = 100
    drop_rate = 0.75

    data_path = "/data1/xiuwen/twitter/tweet2020/tweet-without-conversation/"
    train_data = "train_repeat_post.txt"
    test_data = "test_post.txt"
    path_to_glove_file = "/data1/xiuwen/glove.twitter.27B.200d.txt"

    train_data = read_file(data_path + train_data)
    test_data = read_file(data_path + test_data)
    # train_label = read_file(data_path + "train_tag.txt")
    # test_label = read_file(data_path + "test_tag.txt")

    # vectorize words
    vectorizer = TextVectorization(max_tokens=30000, output_sequence_length=seq_length)
    text_ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
    vectorizer.adapt(text_ds)

    # get mapping from words to indices
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    num_words = len(voc) + 2

    # load pre-trained word embedding
    embeddings_index = load_word_vector(path_to_glove_file)

    # Prepare embedding matrix
    embedding_matrix = build_matrix()

    # one hot encoding for labels
    tags_train, index_tag_mapping = process_labels()
    num_tags = len(index_tag_mapping)

    # prepare data (get topic)
    texts_train = vectorizer(np.array([[s] for s in train_data])).numpy()
    texts_test = vectorizer(np.array([[s] for s in test_data])).numpy()
    # load topic model result
    topics_train_raw = pkl.load(open(data_path + "topic_model_train_repeat.pkl", "rb"))
    topics_train = np.array([np.array([i[1] for i in j]) for j in topics_train_raw])
    topics_test_raw = pkl.load(open(data_path + "topic_model_test.pkl", "rb"))
    topics_test = np.array([np.array([i[1] for i in j]) for j in topics_test_raw])

    myModel = modelDef()
    history = myModel.fit(x=[texts_train, topics_train],
                          y=tags_train,
                          batch_size=batch_size,
                          epochs=30,
                          verbose=1, )
    y_pred = myModel.predict(x=[texts_test, topics_test])
    translate(y_pred, index_tag_mapping)
    # acc, precision, recall, f1 = evaluation(tags_test, y_pred, top_K)
