## basic baseline MF_BPR

import tensorflow as tf
from utils import *

class model_PRM(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'PRM'
        self.pxtr_dim = para['PXTR_DIM']
        self.item_dim = para['ITEM_DIM']
        self.lr = para['LR']
        self.optimizer = para['OPTIMIZER']
        self.n_items = data['item_num']
        self.n_pxtr_bins = para['PXTR_BINS']
        self.max_len = para['CANDIDATE_ITEM_LIST_LENGTH']
        self.pxtr_list = para['PXTR_LIST']

        ## 1 placeholder
        # [-1, max_len]
        #   label
        self.click_label_list = tf.placeholder(tf.int32, shape=[None, self.max_len], name='click_label_list')
        self.like_label_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='like_label_list')
        self.follow_label_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='follow_label_list')
        self.comment_label_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='comment_label_list')
        self.forward_label_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='forward_label_list')
        self.longview_label_list = tf.placeholder(tf.float32, shape=[None, self.max_len], name='longview_label_list')
        self.real_length = tf.placeholder(tf.int32, shape=(None,), name='real_length')
        # self.is_train = tf.placeholder(tf.bool, shape=[], name='is_train')
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

        # 2 reshape
        self.click_label_list_re = tf.reshape(self.click_label_list, [-1, self.max_len])
        self.like_label_list_re = tf.reshape(self.like_label_list, [-1, self.max_len])
        self.follow_label_list_re = tf.reshape(self.follow_label_list, [-1, self.max_len])
        self.comment_label_list_re = tf.reshape(self.comment_label_list, [-1, self.max_len])
        self.forward_label_list_re = tf.reshape(self.forward_label_list, [-1, self.max_len])
        self.longview_label_list_re = tf.reshape(self.longview_label_list, [-1, self.max_len])
        self.real_length_re = tf.reshape(self.real_length, [-1, 1])
        #   pxtr emb
        self.pltr_list = tf.reshape(self.like_pxtr_list, [-1, self.max_len])
        self.pwtr_list = tf.reshape(self.follow_pxtr_list, [-1, self.max_len])
        self.pcmtr_list = tf.reshape(self.comment_pxtr_list, [-1, self.max_len])
        self.pftr_list = tf.reshape(self.forward_pxtr_list, [-1, self.max_len])
        self.plvtr_list = tf.reshape(self.longview_pxtr_list, [-1, self.max_len])
        #   pxtr dense
        self.pltr_dense_list = tf.reshape(self.like_pxtr_dense_list, [-1, self.max_len, 1])
        self.pwtr_dense_list = tf.reshape(self.follow_pxtr_dense_list, [-1, self.max_len, 1])
        self.pcmtr_dense_list = tf.reshape(self.comment_pxtr_dense_list, [-1, self.max_len, 1])
        self.pftr_dense_list = tf.reshape(self.forward_pxtr_dense_list, [-1, self.max_len, 1])
        self.plvtr_dense_list = tf.reshape(self.longview_pxtr_dense_list, [-1, self.max_len, 1])

        # 3 define trainable parameters
        self.pltr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pltr_embeddings_table')
        self.pwtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pwtr_embeddings_table')
        self.pcmtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pcmtr_embeddings_table')
        self.pftr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pftr_embeddings_table')
        self.plvtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='plvtr_embeddings_table')

        # 4 lookup
        #   1) [-1, self.max_len, 1] -> [bs*max_len]
        self.pltr_list = tf.reshape(self.pltr_list, [-1])  
        self.pwtr_list = tf.reshape(self.pwtr_list, [-1])
        self.pcmtr_list = tf.reshape(self.pcmtr_list, [-1])
        self.pftr_list = tf.reshape(self.pftr_list, [-1])
        self.plvtr_list = tf.reshape(self.plvtr_list, [-1])

        #   2) [bs*max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.nn.embedding_lookup(self.pltr_embeddings_table, self.pltr_list)
        self.pwtr_list_embeddings = tf.nn.embedding_lookup(self.pwtr_embeddings_table, self.pwtr_list)
        self.pcmtr_list_embeddings = tf.nn.embedding_lookup(self.pcmtr_embeddings_table, self.pcmtr_list)
        self.pftr_list_embeddings = tf.nn.embedding_lookup(self.pftr_embeddings_table, self.pftr_list)
        self.plvtr_list_embeddings = tf.nn.embedding_lookup(self.plvtr_embeddings_table, self.plvtr_list)

        #   3) [-1, max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.reshape(self.pltr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pwtr_list_embeddings = tf.reshape(self.pwtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pcmtr_list_embeddings = tf.reshape(self.pcmtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pftr_list_embeddings = tf.reshape(self.pftr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.plvtr_list_embeddings = tf.reshape(self.plvtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])

        # 5 start ---------------------
        # [-1, max_len, pxtr_dim*5]
        pxtr_input = tf.concat([self.pltr_list_embeddings, self.pwtr_list_embeddings, self.pcmtr_list_embeddings, 
                                self.pftr_list_embeddings, self.plvtr_list_embeddings], -1)
        # [-1, max_len, 5]
        pxtr_dense_input = tf.concat([self.pltr_dense_list, self.pwtr_dense_list, self.pcmtr_dense_list,
                                      self.pftr_dense_list, self.plvtr_dense_list], -1)

        #   5.2 dropout
        mask = tf.ones_like(pxtr_dense_input)   # [-1, max_len, 5]
        mask = tf.nn.dropout(mask, self.keep_prob)
        mask = tf.expand_dims(mask, -1) # [-1, max_len, 5, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list), self.pxtr_dim]) # [-1, max_len, pxtr_dim*5]->[-1, max_len, 5, pxtr_dim]->
        pxtr_input = pxtr_input * mask           # [-1, max_len, 5, pxtr_dim] * [-1, max_len, 5, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list) * self.pxtr_dim])

        #   5.5 transformer
        with tf.name_scope("sab1"):
            m_size_apply = 32
            head_num = 1
            output_size = self.pxtr_dim
            col = pxtr_input.get_shape()[2]
            mask = tf.sequence_mask(self.real_length_re, maxlen=self.max_len, dtype=tf.float32)
            mask = tf.reshape(mask, [-1, self.max_len])
            pxtr_input = linear_set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="li_trans_encoder", mask=mask,
                col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, m_size=m_size_apply)  # [-1, max_len, nh*pxtr_dim]
            pxtr_input = tf.layers.dense(pxtr_input, output_size, name='realshow_predict_mlp')
            pxtr_input = CommonLayerNorm(pxtr_input, scope='ln_encoder')  # [-1, max_len, pxtr_dim]
            logits = tf.reduce_sum(pxtr_input, axis=2)   # [-1, max_len]
            self.pred = tf.nn.sigmoid(logits)                 # [-1, max_len]
            print("self.pred=", self.pred)
        
        #   5.5 loss
        mask_data = tf.sequence_mask(lengths=self.real_length_re, maxlen=self.max_len)         #序列长度mask
        mask_data = tf.reshape(tf.cast(mask_data, dtype=tf.int32), [-1, self.max_len])
        self.loss = tf.losses.log_loss(self.click_label_list_re, self.pred, weights=mask_data, reduction="weighted_mean")     # loss [-1, max_len]

        #   5.6 optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        #   5.7 update parameters
        self.updates = self.opt.minimize(self.loss)
        print("self.updates=", self.updates)
