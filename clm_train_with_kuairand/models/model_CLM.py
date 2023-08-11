## basic baseline MF_BPR

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn

class model_CLM(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'CLM'
        self.pxtr_dim = para['PXTR_DIM']
        self.item_dim = para['ITEM_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        # self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.n_pxtr_bins = para['PXTR_BINS']
        self.max_len = para['CANDIDATE_ITEM_LIST_LENGTH']
        self.pxtr_list = ['pltr', 'pwtr', 'pcmtr', 'pftr', 'plvtr']
        self.e = 0.1 ** 10
        bin_num = 10000
        if_pxtr_interaction = False
        if_add_position = False

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
        self.item_embeddings_table = tf.Variable(tf.random_normal([self.n_items, self.item_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings_table')
        self.pltr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pltr_embeddings_table')
        self.pwtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pwtr_embeddings_table')
        self.pcmtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pcmtr_embeddings_table')
        self.pftr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='pftr_embeddings_table')
        self.plvtr_embeddings_table = tf.Variable(tf.random_normal([self.n_pxtr_bins, self.pxtr_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='plvtr_embeddings_table')

        # 4 lookup
        self.item_list_re = tf.reshape(self.item_list_re, [-1])  # [-1, max_len] -> [bs*max_len]
        self.item_list_embeddings = tf.nn.embedding_lookup(self.item_embeddings_table, self.item_list_re)  # [bs*max_len, item_dim]
        self.item_list_embeddings = tf.reshape(self.item_list_embeddings, [-1, self.max_len, self.item_dim])  #[-1, max_len, item_dim]

        # 1) [-1, self.max_len, 1] -> [bs*max_len]
        self.pltr_list = tf.reshape(self.pltr_list, [-1])  
        self.pwtr_list = tf.reshape(self.pwtr_list, [-1])
        self.pcmtr_list = tf.reshape(self.pcmtr_list, [-1])
        self.pftr_list = tf.reshape(self.pftr_list, [-1])
        self.plvtr_list = tf.reshape(self.plvtr_list, [-1])

        # 2) [bs*max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.nn.embedding_lookup(self.pltr_embeddings_table, self.pltr_list)
        self.pwtr_list_embeddings = tf.nn.embedding_lookup(self.pwtr_embeddings_table, self.pwtr_list)
        self.pcmtr_list_embeddings = tf.nn.embedding_lookup(self.pcmtr_embeddings_table, self.pcmtr_list)
        self.pftr_list_embeddings = tf.nn.embedding_lookup(self.pftr_embeddings_table, self.pftr_list)
        self.plvtr_list_embeddings = tf.nn.embedding_lookup(self.plvtr_embeddings_table, self.plvtr_list)

        # 3) [-1, max_len, pxtr_dim]
        self.pltr_list_embeddings = tf.reshape(self.pltr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pwtr_list_embeddings = tf.reshape(self.pwtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pcmtr_list_embeddings = tf.reshape(self.pcmtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.pftr_list_embeddings = tf.reshape(self.pftr_list_embeddings, [-1, self.max_len, self.pxtr_dim])
        self.plvtr_list_embeddings = tf.reshape(self.plvtr_list_embeddings, [-1, self.max_len, self.pxtr_dim])

        # 5 start ---------------------
        item_input = self.item_list_embeddings[:, :, 16:]  # [-1, max_len, 48]
        # [-1, max_len, pxtr_dim*5]
        pxtr_input = tf.concat([self.pltr_list_embeddings, self.pwtr_list_embeddings, self.pcmtr_list_embeddings, 
                                self.pftr_list_embeddings, self.plvtr_list_embeddings], -1)
        # [-1, max_len, 5]
        pxtr_dense_input = tf.concat([self.pltr_dense_list, self.pwtr_dense_list, self.pcmtr_dense_list, 
                                      self.pftr_dense_list, self.plvtr_dense_list], -1)

        # 5.1 train item bias of each pxtr
        bias_init = tf.constant_initializer([-6.02140376, - 6.31137081, - 6.96401465, - 6.22389044, 0.92653726])  # initialize bias
        pxtr_item_bias_logits = tf.layers.dense(self.item_list_embeddings[:, :, 0: 16], len(self.pxtr_list), bias_initializer=bias_init, name='pxtr_mlp')   # predict pxtr with item features
        pxtr_item_bias_pred = tf.sigmoid(pxtr_item_bias_logits)
        self.loss_pxtr_bias = tf.losses.log_loss(pxtr_dense_input, pxtr_item_bias_pred, reduction="weighted_mean") # # [-1, max_len, 5]

        # get unbias pxtr; remap and lookup emb
        pxtr_dense_input = tf.clip_by_value(pxtr_dense_input, 0, 1)     # clip pxtr dense value for safty
        pxtr_unbias = pxtr_dense_input / (tf.stop_gradient(pxtr_item_bias_pred) + 0.1 ** 10)   # equals to pxtr_dense_input - pxtr_item_bias_pred
        pxtr_normalize = pxtr_unbias / (pxtr_unbias + 1)       # use x/(1+x) to map (0, inf) to (0, 1)
        pxtr_index = tf.cast(tf.floor(pxtr_normalize * bin_num, name='pxtr_index'), dtype=tf.int64)     # get index
        pxtr_index = tf.clip_by_value(pxtr_index, 0, bin_num - 1)       # clip value for deploy
        pxtr_index = pxtr_index + tf.constant(list(range(len(self.pxtr_list))), dtype=tf.int64) * bin_num    # pctr: index~[0, bin_num), plvtr: index~[bin_num, 2 * bin_num)...
        pxtr_unbias_emb_matrix = tf.get_variable('pxtr_unbias_emb', [bin_num * len(self.pxtr_list), self.pxtr_dim])   # define emb matrix  [10000*5, pxtr_dim]
        pxtr_unbias_emb = tf.nn.embedding_lookup(pxtr_unbias_emb_matrix, pxtr_index)  # [-1, max_len, 5, pxtr_dim]

        # 5.2 dropout
        mask = tf.ones_like(pxtr_dense_input)   # [-1, max_len, 5]
        mask = tf.nn.dropout(mask, self.keep_prob)
        mask = tf.expand_dims(mask, -1) # [-1, max_len, 5, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list), self.pxtr_dim]) # [-1, max_len, pxtr_dim*5]->[-1, max_len, 5, pxtr_dim]->
        pxtr_input = pxtr_input * mask           # [-1, max_len, 5, pxtr_dim] * [-1, max_len, 5, 1]
        pxtr_unbias_emb = pxtr_unbias_emb * mask # [-1, max_len, 5, pxtr_dim] * [-1, max_len, 5, 1]
        pxtr_input = tf.reshape(pxtr_input, [-1, self.max_len, len(self.pxtr_list) * self.pxtr_dim])
        
        pxtr_unbias_input = tf.reshape(pxtr_unbias_emb, [-1, self.max_len, len(self.pxtr_list) * self.pxtr_dim])   # concat embs of pxtr
        
        # 5.3 add position_emb, [-1, max_len, pxtr_dim*5]
        if if_add_position:
            pxtr_input = self.add_position_emb(query_input=pxtr_input, pxtr_dense=pxtr_dense_input, seq_length=self.max_len,
                                               pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, name="biased")
        if if_pxtr_interaction:
            pxtr_input += self.pxtr_transformer(pxtr_input, listwise_len=self.max_len, pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, name='pxtr')
            pxtr_unbias_input += self.pxtr_transformer(pxtr_unbias_input, listwise_len=self.max_len, pxtr_num=len(self.pxtr_list), dim=self.pxtr_dim, name='unbiased_pxtr')
        
        #   5.4 pxtr_input  [-1, max_len, 48 + pxtr_dim*5 + pxtr_dim*5]
        # pxtr_input = tf.concat([item_input, pxtr_input, pxtr_unbias_input], -1)
        pxtr_input = tf.concat([item_input, pxtr_input], -1)

        #   5.5 transformer
        with tf.name_scope("sab1"):
            linear_flag = True
            m_size_apply = 32
            head_num = 1
            layer_num = 0
            output_size = self.pxtr_dim
            col = pxtr_input.get_shape()[2]

            mask = tf.sequence_mask(self.real_length_re, maxlen=self.max_len, dtype=tf.float32)
            mask = tf.reshape(mask, [-1, self.max_len])

            if linear_flag:
                pxtr_input = self.linear_set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="li_trans_encoder", mask=mask,
                    col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, m_size=m_size_apply, iter_num=layer_num)  # [-1, max_len, nh*pxtr_dim]
            else:
                pxtr_input = self.set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="trans_encoder", mask=mask,
                    col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, mask_flag_k=True)
            pxtr_input = tf.layers.dense(pxtr_input, output_size, name='realshow_predict_mlp')
            pxtr_input = self.CommonLayerNorm(pxtr_input, scope='ln_encoder')  # [-1, max_len, pxtr_dim]
            logits = tf.reduce_sum(pxtr_input, axis=2)   # [-1, max_len]
            self.pred = tf.nn.sigmoid(logits)                 # [-1, max_len]
            print("self.pred=", self.pred)
            min_len = tf.reduce_min(self.real_length_re)
            
            self.loss_sim_order = self.sim_order_reg(logits, pxtr_dense_input, para['pxtr_weight'], min_len)

            # choose use or not-use Transformer 
            col = pxtr_input.get_shape()[2]
            if linear_flag:
                pxtr_input = self.linear_set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="li_trans_decoder", mask=mask,
                    col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, m_size=m_size_apply, iter_num=0)  # [-1, listwise_len, nh*dim]
            else:
                pxtr_input = self.set_attention_block(query_input=pxtr_input, action_list_input=pxtr_input, name="trans_decoder", mask=mask,
                    col=col, nh=head_num, action_item_size=col, att_emb_size=output_size, mask_flag_k=True)
            
            pxtr_input = tf.layers.dense(pxtr_input, len(self.pxtr_list), name='pxtr_predict_mlp')
            pxtr_input = self.CommonLayerNorm(pxtr_input, scope='ln_decoder')
            pxtr_pred = tf.nn.sigmoid(pxtr_input)
            self.loss_pxtr_reconstruct = tf.losses.log_loss(pxtr_dense_input, pxtr_pred, reduction="weighted_mean")
        
        #   5.5 loss
        mask_data = tf.sequence_mask(lengths=self.real_length_re, maxlen=self.max_len)         #序列长度mask
        mask_data = tf.reshape(tf.cast(mask_data, dtype=tf.int32), [-1, self.max_len])
        self.loss_click = tf.losses.log_loss(self.click_label_list_re, self.pred, mask_data, reduction="weighted_mean")     # loss [-1, max_len]
        self.loss = para['exp_weight'] * self.loss_click + \
                    para['sim_order_weight'] * self.loss_sim_order + \
                    para['pxtr_reconstruct_weight'] * self.loss_pxtr_reconstruct + \
                    para['bias_weight'] * self.loss_pxtr_bias

        #   5.6 optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        #   5.7 update parameters
        self.updates = self.opt.minimize(self.loss)
        print("self.updates=", self.updates)

    def inner_product(self, users, items):
        scores = tf.reduce_sum(tf.multiply(users, items), axis=1)
        return scores

    def bpr_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def cross_entropy_loss(self, pos_scores, neg_scores):
        maxi = tf.log(tf.nn.sigmoid(pos_scores)) + tf.log(1 - tf.nn.sigmoid(neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def regularization(self, reg_list):
        reg = 0
        for para in reg_list: reg += tf.nn.l2_loss(para)
        return reg

    def linear_set_attention_block(self, query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, m_size=32, iter_num=0):
        ## poly encoder
        with tf.name_scope(name):
            I = tf.get_variable(name + "_i_trans_matrix",(1, m_size, col), initializer=tf.truncated_normal_initializer(stddev=5.0)) # [-1, m_size, col]
            I = tf.tile(I, [tf.shape(query_input)[0],1,1])
            H = self.set_attention_block(I, action_list_input, name + "_ele2clus", mask, col, 1, action_item_size, att_emb_size, True, True)    #[-1, m_size, nh*dim]
            # for l in range(iter_num):
            #     H = self.set_attention_block(H, action_list_input, name + "_ele2clus_{}".format(l), mask, att_emb_size, 1, action_item_size, att_emb_size, True, True)  # [-1, m_size, nh*dim]
            for l in range(iter_num):
                H += self.set_attention_block(H, H, name + "_sa_clus2clus_{}".format(l), mask, att_emb_size, 1, att_emb_size, att_emb_size, False, False)
                H = self.CommonLayerNorm(H, scope='ln1_clus2clus_{}'.format(l))
                H += tf.layers.dense(tf.nn.relu(H), att_emb_size, name='ffn_clus2clus_{}'.format(l))
                H = self.CommonLayerNorm(H, scope='ln2_clus2clus_{}'.format(l))
            res = self.set_attention_block(query_input, H, name + "_clus2ele", mask, col, nh, att_emb_size, att_emb_size, True, False)
        return res

    # query_input =》 [-1, list_size_q=1, dim]     k=v=[-1, list_size_k, dim]
    # retun : [-1, list_size_q=1, nh*dim]
    def set_attention_block(self, query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, if_mask=True, mask_flag_k=True):
        with tf.name_scope("mha_" + name):
            batch_size = tf.shape(query_input)[0]
            list_size = tf.shape(query_input)[1]
            list_size_k = tf.shape(action_list_input)[1]
            Q = tf.get_variable(name + '_q_trans_matrix', (col, att_emb_size * nh))
            K = tf.get_variable(name + '_k_trans_matrix', (action_item_size, att_emb_size * nh))
            V = tf.get_variable(name + '_v_trans_matrix', (action_item_size, att_emb_size * nh))

            querys = tf.tensordot(query_input, Q, axes=(-1, 0))
            keys = tf.tensordot(action_list_input, K, axes=(-1, 0))
            values = tf.tensordot(action_list_input, V, axes=(-1, 0))

            querys = tf.stack(tf.split(querys, nh, axis=2))
            keys = tf.stack(tf.split(keys, nh, axis=2))
            values = tf.stack(tf.split(values, nh, axis=2))

            inner_product = tf.matmul(querys, keys, transpose_b=True) / 8.0
            if if_mask:
                trans_mask = tf.tile(tf.expand_dims(mask, axis=0),[nh, 1, 1])
                if mask_flag_k: trans_mask = tf.tile(tf.expand_dims(trans_mask, axis=2), [1,1,list_size,1])
                else: trans_mask = tf.tile(tf.expand_dims(trans_mask, axis=3), [1, 1, 1, list_size_k])
                paddings = tf.ones_like(trans_mask) * (-2 ** 32 + 1)
                inner_product = tf.where(tf.equal(trans_mask, 0), paddings, inner_product)

            normalized_att_scores = tf.nn.softmax(inner_product)
            result = tf.matmul(normalized_att_scores, values)
            result = tf.transpose(result, perm=[1, 2, 0, 3])
            mha_result = tf.reshape(result, [batch_size, list_size, nh * att_emb_size])
        return mha_result


    def CommonLayerNorm(self, inputs,
                    center=True,
                    scale=True,
                    activation_fn=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    begin_norm_axis=1,
                    begin_params_axis=-1,
                    scope=None):
        with variable_scope.variable_scope(
                scope, 'LayerNorm', [inputs], reuse=reuse) as sc:
            inputs = ops.convert_to_tensor(inputs)
            inputs_shape = inputs.shape
            inputs_rank = inputs_shape.ndims
            if inputs_rank is None:
                raise ValueError('Inputs %s has undefined rank.' % inputs.name)
            dtype = inputs.dtype.base_dtype
            if begin_norm_axis < 0:
                begin_norm_axis = inputs_rank + begin_norm_axis
            if begin_params_axis >= inputs_rank or begin_norm_axis >= inputs_rank:
                raise ValueError('begin_params_axis (%d) and begin_norm_axis (%d) '
                            'must be < rank(inputs) (%d)' %
                            (begin_params_axis, begin_norm_axis, inputs_rank))
            params_shape = inputs_shape[begin_params_axis:]
            if not params_shape.is_fully_defined():
                raise ValueError(
                    'Inputs %s: shape(inputs)[%s:] is not fully defined: %s' %
                    (inputs.name, begin_params_axis, inputs_shape))
            # Allocate parameters for the beta and gamma of the normalization.
            beta, gamma = None, None
            if center:
                beta_collections = utils.get_variable_collections(variables_collections,
                                                                    'beta')
                beta = tf.get_variable(
                    'beta',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=init_ops.zeros_initializer(),
                    collections=beta_collections,
                    trainable=trainable)
            if scale:
                gamma_collections = utils.get_variable_collections(
                    variables_collections, 'gamma')
                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    dtype=dtype,
                    initializer=init_ops.ones_initializer(),
                    collections=gamma_collections,
                    trainable=trainable)
            # Calculate the moments on the last axis (layer activations).
            norm_axes = list(range(begin_norm_axis, inputs_rank))
            mean, variance = nn.moments(inputs, norm_axes, keep_dims=True)
            # Compute layer normalization using the batch_normalization function.
            variance_epsilon = 1e-12
            outputs = nn.batch_normalization(
                inputs,
                mean,
                variance,
                offset=beta,
                scale=gamma,
                variance_epsilon=variance_epsilon)
            outputs.set_shape(inputs_shape)
            if activation_fn is not None:
                outputs = activation_fn(outputs)
            return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


    def add_position_emb(self, query_input, pxtr_dense, seq_length, pxtr_num, dim, name):
        with tf.name_scope(name):
            pos_embeddings = tf.get_variable(name+"_pos_embeddings", (seq_length, dim), initializer=tf.truncated_normal_initializer(stddev=5.0))  # [-1, m_size, col]
            position_in_ranking = tf.contrib.framework.argsort(tf.stop_gradient(pxtr_dense), axis=1)
            order = tf.contrib.framework.argsort(position_in_ranking, axis=1)
            pos_emb = tf.gather(pos_embeddings, order)
            return tf.reshape(pos_emb, [-1, seq_length, pxtr_num * dim]) + query_input

    def pxtr_transformer(self, pxtr_input, listwise_len, pxtr_num, dim, name):
        pxtr_input = tf.reshape(pxtr_input, [-1, pxtr_num, dim])
        pxtr_input = self.set_attention_block(pxtr_input, pxtr_input, name + "_pxtr_transformer", 0, dim, 1, dim, dim, False, False)
        return tf.reshape(pxtr_input, [-1, listwise_len, pxtr_num * dim])

    def sigmoid(self, x):
        return 2 * tf.nn.sigmoid(x) - 1

    def sim_order_reg_core(self, seq_1, seq_2, if_norm, length):
        seq_conc = tf.concat([tf.expand_dims(seq_1, -1), tf.expand_dims(seq_2, -1)], -1)
        seq_cut = seq_conc[:, 0: length, :]
        random_index = tf.random.uniform((1, 200), minval=0, maxval=tf.cast(length, dtype=tf.float32))
        random_index = tf.squeeze(tf.cast(tf.floor(random_index), dtype=tf.int64))
        seq_samp = tf.gather(seq_cut, random_index, axis=1)
        seq = tf.reshape(seq_samp, [-1, 2])
        if if_norm:
            seq_mean, seq_var = tf.nn.moments(tf.stop_gradient(seq), axes=0)
            seq_norm = (seq - tf.expand_dims(seq_mean, 0)) / (tf.sqrt(tf.expand_dims(seq_var, 0)) + self.e)
        else:
            seq_norm = seq
        seq_resh = tf.reshape(seq_norm, [-1, 2, 2])
        # attention! sigmoid(x) = 2 * tf.nn.sigmoid(x) - 1
        # TODO: replace with tanh(x)
        reg_loss = tf.multiply(self.sigmoid(seq_resh[:, 0, 0] - seq_resh[:, 1, 0]), self.sigmoid(seq_resh[:, 0, 1] - seq_resh[:, 1, 1]))
        return -tf.reduce_mean(reg_loss)

    def sim_order_reg(self, pred, pxtr, weight, length):
        reg_loss = 0
        for i, w in enumerate(weight):
            reg_loss += w * self.sim_order_reg_core(pred, pxtr[:, :, i], True, length)
            # reg_loss += weaken_bad_pxtr_weight * w * self.sim_order_reg_core(pred, -1 / (pxtr[:, :, i] + e), True, length)
        return reg_loss


