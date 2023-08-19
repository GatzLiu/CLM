## basic baseline MF_BPR

import tensorflow as tf
from utils import *

class model_CLM(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'CLM'
        self.pxtr_dim = para['PXTR_DIM']
        self.item_dim = para['ITEM_DIM']
        self.lr = para['LR']
        self.optimizer = para['OPTIMIZER']
        self.n_items = data['item_num']
        self.n_pxtr_bins = para['PXTR_BINS']
        self.max_len = para['CANDIDATE_ITEM_LIST_LENGTH']
        self.pxtr_list = para['PXTR_LIST']
        bin_num = 10000
        decay = 0.01
        if_pxtr_interaction = False
        if_add_position = True

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

        # 1) [-1, self.max_len, 1] -> [bs*max_len]
        self.pltr_list = tf.reshape(self.pltr_list, [-1])  
        self.pwtr_list = tf.reshape(self.pwtr_list, [-1])
        self.pftr_list = tf.reshape(self.pftr_list, [-1])

        # 2) [bs*max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.nn.embedding_lookup(self.pltr_embeddings_table, self.pltr_list)
        self.pwtr_list_embeddings = tf.nn.embedding_lookup(self.pwtr_embeddings_table, self.pwtr_list)
        self.pftr_list_embeddings = tf.nn.embedding_lookup(self.pftr_embeddings_table, self.pftr_list)

        # 3) [-1, max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.reshape(self.pltr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pwtr_list_embeddings = tf.reshape(self.pwtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pftr_list_embeddings = tf.reshape(self.pftr_list_embeddings, [-1, self.max_len, self.pxtr_dim])

        # 5 start ---------------------
        item_input = self.item_list_embeddings[:, :, 16:]  # [-1, max_len, 48]
        # [-1, max_len, pxtr_dim*3]
        pxtr_input = tf.concat([self.pltr_list_embeddings, self.pwtr_list_embeddings, self.pftr_list_embeddings], -1)
        # [-1, max_len, 3]
        pxtr_dense_input = tf.concat([self.pltr_dense_list, self.pwtr_dense_list, self.pftr_dense_list], -1)

        # 5.1 train item bias of each pxtr
        bias_init = tf.constant_initializer([-6.02140376, - 6.31137081, - 6.96401465])  # initialize bias
        pxtr_item_bias_logits = tf.layers.dense(self.item_list_embeddings[:, :, 0: 16], len(self.pxtr_list), bias_initializer=bias_init, name='pxtr_mlp')   # predict pxtr with item features
        pxtr_item_bias_pred = tf.sigmoid(pxtr_item_bias_logits)
        self.loss_pxtr_bias = tf.losses.log_loss(pxtr_dense_input, pxtr_item_bias_pred, reduction="weighted_mean") # # [-1, max_len, 3]

        # get unbias pxtr; remap and lookup emb
        pxtr_dense_input = tf.clip_by_value(pxtr_dense_input, 0, 1)     # clip pxtr dense value for safty
        pxtr_unbias = pxtr_dense_input / (tf.stop_gradient(pxtr_item_bias_pred) + 0.1 ** 10)   # equals to pxtr_dense_input - pxtr_item_bias_pred
        pxtr_normalize = pxtr_unbias / (pxtr_unbias + 1)       # use x/(1+x) to map (0, inf) to (0, 1)
        pxtr_index = tf.cast(tf.floor(pxtr_normalize * bin_num, name='pxtr_index'), dtype=tf.int64)     # get index
        pxtr_index = tf.clip_by_value(pxtr_index, 0, bin_num - 1)       # clip value for deploy
        pxtr_index = pxtr_index + tf.constant(list(range(len(self.pxtr_list))), dtype=tf.int64) * bin_num    # pctr: index~[0, bin_num), plvtr: index~[bin_num, 2 * bin_num)...
        pxtr_unbias_emb_matrix = tf.get_variable('pxtr_unbias_emb', [bin_num * len(self.pxtr_list), self.pxtr_dim])   # define emb matrix  [10000*5, pxtr_dim]
        pxtr_unbias_emb = tf.nn.embedding_lookup(pxtr_unbias_emb_matrix, pxtr_index)  # [-1, max_len, 3, pxtr_dim]

        # 5.2 dropout
        mask = tf.ones_like(pxtr_input)   # [-1, max_len, 3]
        mask = tf.nn.dropout(mask, self.keep_prob)
        mask = tf.expand_dims(mask, -1) # [-1, max_len, 3, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list), self.pxtr_dim]) # [-1, max_len, pxtr_dim*3]->[-1, max_len, 3, pxtr_dim]->
        pxtr_input = pxtr_input * mask           # [-1, max_len, 3, pxtr_dim] * [-1, max_len, 3, 1]
        pxtr_unbias_emb = pxtr_unbias_emb * mask # [-1, max_len, 3, pxtr_dim] * [-1, max_len, 3, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list) * self.pxtr_dim])
        
        pxtr_unbias_input = tf.reshape(pxtr_unbias_emb, [-1, self.max_len, len(self.pxtr_list) * self.pxtr_dim])   # concat embs of pxtr
        
        # 5.3 add position_emb, [-1, max_len, pxtr_dim*3]
        if if_add_position:
            pxtr_input = add_position_emb(query_input=pxtr_input, pxtr_dense=pxtr_dense_input, seq_length=self.max_len,
                                          pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, decay=decay, name="biased")
        if if_pxtr_interaction:
            pxtr_input += pxtr_transformer(pxtr_input, listwise_len=self.max_len, pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, name='pxtr')
            pxtr_unbias_input += pxtr_transformer(pxtr_unbias_input, listwise_len=self.max_len, pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, name='unbiased_pxtr')
        
        #   5.4 pxtr_input  [-1, max_len, 48 + pxtr_dim*5 + pxtr_dim*5]
        if para['if_debias']: pxtr_input = tf.concat([item_input, pxtr_input, decay * pxtr_unbias_input], -1)

        #   5.5 transformer
        with tf.name_scope("sab1"):
            m_size_apply = 32
            head_num = 1
            output_size = self.pxtr_dim
            col = pxtr_input.get_shape()[2]
            mask = tf.sequence_mask(self.real_length_re, maxlen=self.max_len, dtype=tf.float32)
            mask = tf.reshape(mask, [-1, self.max_len])
            pxtr_input = linear_set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="li_trans_encoder", mask=mask,
                col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, m_size=m_size_apply, iter_num=para['layer_num'])  # [-1, max_len, nh*pxtr_dim]
            pxtr_input = tf.layers.dense(pxtr_input, output_size, name='realshow_predict_mlp')
            pxtr_input = CommonLayerNorm(pxtr_input, scope='ln_encoder')  # [-1, max_len, pxtr_dim]
            logits = tf.reduce_sum(pxtr_input, axis=2)   # [-1, max_len]
            self.pred = tf.nn.sigmoid(logits)                 # [-1, max_len]

        #   5.5 loss
        mask_data = tf.sequence_mask(lengths=self.real_length_re, maxlen=self.max_len)         #序列长度mask
        mask_data = tf.reshape(tf.cast(mask_data, dtype=tf.int32), [-1, self.max_len])
        self.loss_click = tf.losses.log_loss(self.click_label_list_re, self.pred, mask_data, reduction="weighted_mean")     # loss [-1, max_len]
        self.loss = para['exp_weight'] * self.loss_click + \
                    para['bias_weight'] * self.loss_pxtr_bias

        #   5.6 optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        #   5.7 update parameters
        self.updates = self.opt.minimize(self.loss)
