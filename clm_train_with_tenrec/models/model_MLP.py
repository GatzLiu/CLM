## basic baseline MF_BPR

import tensorflow as tf
from utils import *

class model_MLP(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'MLP'
        self.pxtr_dim = para['PXTR_DIM']
        self.item_dim = para['ITEM_DIM']
        self.lr = para['LR']
        self.optimizer = para['OPTIMIZER']
        self.n_items = data['item_num']
        self.n_pxtr_bins = para['PXTR_BINS']
        self.max_len = para['CANDIDATE_ITEM_LIST_LENGTH']
        self.pxtr_list = para['PXTR_LIST']
        self.e = 0.1 ** 10

        ## 1 placeholder
        # [-1, max_len]
        self.item_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='item_list')   # [-1, max_len]
        #   label
        self.click_label_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='click_label_list')
        self.real_length = tf.placeholder(tf.int32, shape=(None,), name='real_length')
        # self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')
        #   pxtr emb feature
        self.like_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='like_pxtr_list')   # bin
        self.follow_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='follow_pxtr_list')
        self.forward_pxtr_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='forward_pxtr_list')
        #   pxtr dense feature
        self.like_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='like_pxtr_dense_list')
        self.follow_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='follow_pxtr_dense_list')
        self.forward_pxtr_dense_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='forward_pxtr_dense_list')
        print ("self.item_list: ", self.item_list)

        # 2 reshape
        self.item_list_re = tf.reshape(self.item_list, [-1, self.max_len])
        self.click_label_list_re = tf.reshape(self.click_label_list, [-1, self.max_len])
        self.real_length_re = tf.reshape(self.real_length, [-1, 1])
        #   pxtr emb
        self.pltr_list = tf.reshape(self.like_pxtr_list, [-1, self.max_len])
        self.pwtr_list = tf.reshape(self.follow_pxtr_list, [-1, self.max_len])
        self.pftr_list = tf.reshape(self.forward_pxtr_list, [-1, self.max_len])
        #   pxtr dense
        self.pltr_dense_list = tf.reshape(self.like_pxtr_dense_list, [-1, self.max_len, 1])
        self.pwtr_dense_list = tf.reshape(self.follow_pxtr_dense_list, [-1, self.max_len, 1])
        self.pftr_dense_list = tf.reshape(self.forward_pxtr_dense_list, [-1, self.max_len, 1])

        # 3 define trainable parameters
        self.item_embeddings_table = tf.Variable(tf.random_normal([self.n_items, self.item_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_table')
        self.pltr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pltr_embeddings_table')
        self.pwtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pwtr_embeddings_table')
        self.pftr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pftr_embeddings_table')

        # 4 lookup
        self.item_list_re = tf.reshape(self.item_list_re, [-1])  # [-1, max_len] -> [bs*max_len]
        self.item_list_embeddings = tf.nn.embedding_lookup(self.item_embeddings_table, self.item_list_re)  # [bs*max_len, item_dim]
        self.item_list_embeddings = tf.reshape(self.item_list_embeddings, [-1, self.max_len, self.item_dim])  #[-1, max_len, item_dim]

        #   1) [-1, self.max_len, 1] -> [bs*max_len]
        self.pltr_list = tf.reshape(self.pltr_list, [-1])  
        self.pwtr_list = tf.reshape(self.pwtr_list, [-1])
        self.pftr_list = tf.reshape(self.pftr_list, [-1])

        #   2) [bs*max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.nn.embedding_lookup(self.pltr_embeddings_table, self.pltr_list)
        self.pwtr_list_embeddings = tf.nn.embedding_lookup(self.pwtr_embeddings_table, self.pwtr_list)
        self.pftr_list_embeddings = tf.nn.embedding_lookup(self.pftr_embeddings_table, self.pftr_list)

        #   3) [-1, max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.reshape(self.pltr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pwtr_list_embeddings = tf.reshape(self.pwtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pftr_list_embeddings = tf.reshape(self.pftr_list_embeddings, [-1, self.max_len, self.pxtr_dim])


        # 5 start ---------------------
        item_input = self.item_list_embeddings[:, :, 16:]  # [-1, max_len, 48]
        # [-1, max_len, pxtr_dim*5]
        pxtr_input = tf.concat([self.pltr_list_embeddings, self.pwtr_list_embeddings, self.pftr_list_embeddings], -1)
        # [-1, max_len, 5]
        pxtr_dense_input = tf.concat([self.pltr_dense_list, self.pwtr_dense_list, self.pftr_dense_list], -1)

        # pxtr_input = tf.concat([item_input, pxtr_input], -1)
        #   5.5 mlp
        with tf.name_scope("mlp"):
            output_size = self.pxtr_dim
            if para['mode'] == 'LR':
                logits = tf.layers.dense(pxtr_input, 1, name='realshow_predict_lr')
                logits = tf.squeeze(logits, -1)
            if para['mode'] == 'MLP':
                pxtr_input = tf.layers.dense(pxtr_input, output_size, name='realshow_predict_mlp')
                pxtr_input = CommonLayerNorm(pxtr_input, scope='ln_encoder')  # [-1, max_len, pxtr_dim]
                logits = tf.reduce_sum(pxtr_input, axis=2)   # [-1, max_len]
            self.pred = tf.nn.sigmoid(logits)                 # [-1, max_len]
            print("self.pred=", self.pred)
            min_len = tf.reduce_min(self.real_length_re)
            self.loss_sim_order = sim_order_reg(logits, pxtr_dense_input, para['pxtr_weight'], min_len)

            # choose use or not-use Transformer
            pxtr_input = tf.layers.dense(pxtr_input, len(self.pxtr_list), name='pxtr_predict_mlp')
            pxtr_input = CommonLayerNorm(pxtr_input, scope='ln_decoder')
            pxtr_pred = tf.nn.sigmoid(pxtr_input)
            self.loss_pxtr_reconstruct = tf.losses.log_loss(pxtr_dense_input, pxtr_pred, reduction="weighted_mean")
        
        #   5.5 loss
        mask_data = tf.sequence_mask(lengths=self.real_length_re, maxlen=self.max_len)         #序列长度mask
        mask_data = tf.reshape(tf.cast(mask_data, dtype=tf.int32), [-1, self.max_len])
        self.loss_click = tf.losses.log_loss(self.click_label_list_re, self.pred, mask_data, reduction="weighted_mean")     # loss [-1, max_len]
        self.loss = para['exp_weight'] * self.loss_click + \
                    para['sim_order_weight'] * self.loss_sim_order + \
                    para['pxtr_reconstruct_weight'] * self.loss_pxtr_reconstruct
        self.loss_pxtr_bias = tf.constant(0)

        #   5.6 optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        #   5.7 update parameters
        self.updates = self.opt.minimize(self.loss)
        print("self.updates=", self.updates)
