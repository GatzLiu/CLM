## basic baseline MF_BPR

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn

class model_MUL(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'MUL'
        self.pxtr_dim = para['PXTR_DIM']
        self.item_dim = para['ITEM_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.optimizer = para['OPTIMIZER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        # self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.n_pxtr_bins = para['PXTR_BINS']
        self.max_len = para['CANDIDATE_ITEM_LIST_LENGTH']
        self.pxtr_list = ['pltr', 'pwtr', 'pcmtr', 'pftr', 'plvtr']
        self.e = 0.1 ** 10

        ## 1 placeholder
        # [-1, max_len]
        self.item_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='item_list')   # [-1, max_len]
        #   label
        self.click_label_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='click_label_list')
        self.real_length = tf.placeholder(tf.int32, shape=(None,), name='real_length')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
        #   pxtr emb feature
        self.like_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='like_pxtr_list')   # bin
        self.follow_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='follow_pxtr_list')
        self.comment_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='comment_pxtr_list')
        self.forward_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='forward_pxtr_list')
        self.longview_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='longview_pxtr_list')
        #   pxtr dense feature
        self.like_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='like_pxtr_dense_list')
        self.follow_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='follow_pxtr_dense_list')
        self.comment_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='comment_pxtr_dense_list')
        self.forward_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='forward_pxtr_dense_list')
        self.longview_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='longview_pxtr_dense_list')
        print ("self.item_list: ", self.item_list)

        # 2 reshape
        self.item_list_re = tf.reshape(self.item_list, [-1, self.max_len])
        self.click_label_list_re = tf.reshape(self.click_label_list, [-1, self.max_len])
        self.real_length_re = tf.reshape(self.real_length, [-1, 1])
        #   pxtr dense
        self.pltr_dense_list = tf.reshape(self.like_pxtr_dense_list, [-1, self.max_len])
        self.pwtr_dense_list = tf.reshape(self.follow_pxtr_dense_list, [-1, self.max_len])
        self.pcmtr_dense_list = tf.reshape(self.comment_pxtr_dense_list, [-1, self.max_len])
        self.pftr_dense_list = tf.reshape(self.forward_pxtr_dense_list, [-1, self.max_len])
        self.plvtr_dense_list = tf.reshape(self.longview_pxtr_dense_list, [-1, self.max_len])

        logits = (1 + self.pltr_dense_list / para['beta_ltr']) ** para['alpha_ltr']
        logits *= (1 + self.pwtr_dense_list / para['beta_wtr']) ** para['alpha_wtr']
        logits *= (1 + self.pcmtr_dense_list / para['beta_cmtr']) ** para['alpha_cmtr']
        logits *= (1 + self.pftr_dense_list / para['beta_ftr']) ** para['alpha_ftr']
        logits *= (1 + self.plvtr_dense_list / para['beta_lvtr']) ** para['alpha_lvtr']
        self.pred = tf.nn.sigmoid(logits)

        # 3 define trainable parameters
        self.item_embeddings_table = tf.Variable(tf.random_normal([self.n_items, self.item_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_table')
        self.item_list_re = tf.reshape(self.item_list_re, [-1])  # [-1, max_len] -> [bs*max_len]
        self.item_list_embeddings = tf.nn.embedding_lookup(self.item_embeddings_table, self.item_list_re)  # [bs*max_len, item_dim]
        self.item_list_embeddings = tf.reshape(self.item_list_embeddings, [-1, self.max_len, self.item_dim])  #[-1, max_len, item_dim]
        
        #   5.5 loss
        self.loss = tf.nn.l2_loss(self.item_list_embeddings)
        self.loss_click = tf.constant(0)
        self.loss_sim_order = tf.constant(0)
        self.loss_pxtr_reconstruct = tf.constant(0)
        self.loss_pxtr_bias = tf.constant(0)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss)
        print("self.updates=", self.updates)
