"""Implementation of CDAE.
TODO
"""

import tensorflow as tf
import time
import numpy as np
import math

class CDAE(object):
    def __init__(self, 
                 sess: tf.compat.v1.Session, 
                 num_user: int, 
                 num_item: int, 
                 learning_rate: float = 0.01, 
                 reg_rate: float = 0.01, 
                 epoch: int = 500, 
                 batch_size: int = 100,
                 verbose: bool = False, 
                 t: int = 1, 
                 display_step: int = 1000, 
                 path: str = ""):

        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.T = t
        self.display_step = display_step
        self.path = path

        self.user_id = None
        self.corrupted_rating_matrix = None
        self.rating_matrix = None
        self.corruption_level = None
        self.layer_2 = None
        self.loss = None
        self.optimizer = None
        self.train_data = None
        self.neg_items = None
        self.num_training = None
        self.total_batch = None
        self.test_data = None
        self.test_users = None
        self.reconstruction = None
        print("You are running CDAE.")

    def build_network(self, 
                      hidden_neuron: int = 500, 
                      corruption_level: int = 0) -> None:

        self.corrupted_rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_item])
        self.rating_matrix = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.num_item])
        self.user_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.corruption_level = corruption_level

        _W = tf.Variable(tf.random.normal([self.num_item, hidden_neuron], stddev=0.01))
        _W_prime = tf.Variable(tf.random.normal([hidden_neuron, self.num_item], stddev=0.01))
        _V = tf.Variable(tf.random.normal([self.num_user, hidden_neuron], stddev=0.01))

        b = tf.Variable(tf.random.normal([hidden_neuron], stddev=0.01))
        b_prime = tf.Variable(tf.random.normal([self.num_item], stddev=0.01))

        layer_1 = tf.sigmoid(tf.matmul(self.corrupted_rating_matrix, _W) + tf.nn.embedding_lookup(params=_V, ids=self.user_id) + b)
        self.layer_2 = tf.sigmoid(tf.matmul(layer_1, _W_prime) + b_prime)

        self.loss = - tf.reduce_sum(
            input_tensor=self.rating_matrix * tf.math.log(self.layer_2) + (1 - self.rating_matrix) * tf.math.log(1 - self.layer_2)) + \
            self.reg_rate * (tf.nn.l2_loss(_W) + tf.nn.l2_loss(_W_prime) + tf.nn.l2_loss(_V) +
                             tf.nn.l2_loss(b) + tf.nn.l2_loss(b_prime))

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def prepare_data(self, 
                     train_data: dict, 
                     test_data: dict) -> None:

        self.train_data = self._data_process(train_data)
        self.neg_items = self._get_neg_items(train_data)
        self.num_training = self.num_user
        self.total_batch = int(self.num_training / self.batch_size)
        self.test_data = test_data
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("Data preparation finished.")

    def train(self) -> None:
        idxs = np.random.permutation(self.num_training)  

        for i in range(self.total_batch):
            start_time = time.time()
            if i == self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size:]
            elif i < self.total_batch - 1:
                batch_set_idx = idxs[i * self.batch_size: (i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={
                self.corrupted_rating_matrix: self._get_corrupted_input(self.train_data[batch_set_idx, :],
                                                                        self.corruption_level),
                self.rating_matrix: self.train_data[batch_set_idx, :],
                self.user_id: batch_set_idx
                })
            if self.verbose and i % self.display_step == 0:
                print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                if self.verbose:
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self) -> None:
        self.reconstruction = self.sess.run(self.layer_2, feed_dict={self.corrupted_rating_matrix: self.train_data,
                                                                     self.user_id: range(self.num_user)})

        evaluate(self)

    def execute(self, 
                train_data: dict, 
                test_data: dict) -> None:

        self.prepare_data(train_data, test_data)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        for epoch in range(self.epochs):
            self.train()
            if epoch % self.T == 0:
                print("Epoch: %04d; " % epoch, end='')
                self.test()

    def save(self, 
             name: str) -> None:

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, self.path + name)

    def predict(self, 
                user_id: int, 
                item_id: int) -> None:
        return np.array(self.reconstruction[np.array(user_id), np.array(item_id)])

    @staticmethod
    def _data_process(data: dict) -> np.matrix:
        return np.asmatrix(data)

    def _get_neg_items(self, 
                       data: dict) -> dict:
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = [k for k, i in enumerate(data[u]) if data[u][k] == 0]

        return neg_items

    @staticmethod
    def _get_corrupted_input(input_train_data: np.number, 
                             corruption_level: np.number) -> np.number:

        return np.random.binomial(n=1, p=1 - corruption_level) * input_train_data

def precision_recall_ndcg_at_k(k: int, 
                               rankedlist: list, 
                               test_matrix: np.matrix) -> float:
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)

def map_mrr_ndcg(rankedlist: list, 
                 test_matrix: np.matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)

def evaluate(self) -> None:
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []

        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)
        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    with open(self.path + "results.txt", "a") as file:
        file.write("------------------------\n")
        file.write("precision@10:" + str(np.mean(p_at_10)) + "\n")
        file.write("recall@10:" + str(np.mean(r_at_10)) + "\n")
        file.write("precision@5:" + str(np.mean(p_at_5)) + "\n")
        file.write("recall@5:" + str(np.mean(r_at_5)) + "\n")
        file.write("map:" + str(np.mean(map)) + "\n")
        file.write("mrr:" + str(np.mean(mrr)) + "\n")
        file.write("ndcg:" + str(np.mean(ndcg)) + "\n")
        file.write("ndcg@5:" + str(np.mean(ndcg_at_5)) + "\n")
        file.write("ndcg@10:" + str(np.mean(ndcg_at_10)) + "\n")