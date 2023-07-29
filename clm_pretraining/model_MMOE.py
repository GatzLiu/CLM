## basic baseline MF_BPR

import tensorflow as tf

class model_MMOE(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'MMOE'
        self.emb_dim = para['EMB_DIM']
        self.lr = para['LR']
        self.lamda = para['LAMDA']
        self.loss_function = para['LOSS_FUNCTION']
        self.optimizer = para['OPTIMIZER']
        self.sampler = para['SAMPLER']
        self.aux_loss_weight = para['AUX_LOSS_WEIGHT']
        self.n_users = data['user_num']
        self.n_items = data['item_num']
        self.max_len = para['ACTION_LIST_MAX_LEN']
        # self.popularity = data['popularity']
        # self.A_hat = data['sparse_propagation_matrix']
        # self.graph_emb = data['graph_embeddings']

        ## placeholder
        self.users = tf.placeholder(tf.int32, shape=(None,)) # index []
        self.items = tf.placeholder(tf.int32, shape=(None,))
        # self.action_list = tf.placeholder(tf.int32, shape=[None, 150]) # [-1, max_len]
        # self.action_list = tf.placeholder(tf.int32) # [-1, max_len]
        self.real_length = tf.placeholder(tf.int32, shape=(None,))
        self.lable_like = tf.placeholder(tf.int32, shape=(None,))
        self.lable_follow = tf.placeholder(tf.int32, shape=(None,))
        self.lable_comment = tf.placeholder(tf.int32, shape=(None,))
        self.lable_forward = tf.placeholder(tf.int32, shape=(None,))
        self.lable_longview = tf.placeholder(tf.int32, shape=(None,))
        # print("0 tf.shape(self.action_list)=", tf.shape(self.action_list))
        # self.action_list = tf.reshape(self.action_list, [-1, 150])

        ## define trainable parameters
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='user_embeddings')
        self.item_embeddings = tf.Variable(tf.random_normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32), name='item_embeddings')
        self.var_list = [self.user_embeddings, self.item_embeddings]


        ## lookup
        self.u_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.users) # [-1, dim]
        self.i_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.items) # [-1, dim]

        # print("1 tf.shape(self.action_list)=", tf.shape(self.action_list))
        # self.action_list = tf.reshape(self.action_list, [-1])  # [-1, max_len] -> [bs*max_len]
        # print("2 tf.shape(self.action_list)=", tf.shape(self.action_list))

        # self.action_list_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.action_list)  # [bs*max_len, dim]
        # self.action_list_embeddings = tf.reshape(self.action_list_embeddings, [-1, self.max_len, self.emb_dim])  #[-1, max_len, dim]

        # reshape
        self.real_length = tf.reshape(self.real_length, [-1, 1])
        self.lable_like = tf.reshape(self.lable_like, [-1, 1])
        self.lable_follow = tf.reshape(self.lable_follow, [-1, 1])
        self.lable_comment = tf.reshape(self.lable_comment, [-1, 1])
        self.lable_forward = tf.reshape(self.lable_forward, [-1, 1])
        self.lable_longview = tf.reshape(self.lable_longview, [-1, 1])

    #     # start ---------------------
    #     mask = tf.sequence_mask(self.real_length, maxlen=self.max_len, dtype=tf.float32)
    #     mask = tf.reshape(mask, [-1, self.max_len]) 

    #    # target_attention
    #     self.i_embeddings = tf.reshape(self.i_embeddings, [-1, 1, self.emb_dim])
    #     # [-1, list_size_q=1, nh*dim]
    #     taget_attention_input = self.set_attention_block(self.i_embeddings, self.action_list_embeddings, name="target_attention", mask=mask, 
    #                             col=self.emb_dim, nh=1, action_item_size=self.emb_dim, att_emb_size=self.emb_dim, mask_flag_k=True)
    #     print("tf.shape(taget_attention_input)=", tf.shape(taget_attention_input))
    #     taget_attention_input = tf.reshape(taget_attention_input, [-1, self.emb_dim])

    #     # mmoe
    #     self.i_embeddings = tf.reshape(self.i_embeddings, [-1, self.emb_dim])
    #     feature_input = tf.concat([self.u_embeddings, self.i_embeddings, taget_attention_input], -1)

    #     feature_input = tf.reshape(feature_input, [-1, 1, self.emb_dim*3])
    #     # [-1, 1, att_emb_size] ** num_tasks
    #     mmoe_output = self.mmoe_layer(feature_input, att_emb_size=32, num_experts=6, num_tasks=5)

    #     # logit
    #     # [-1, 1, 1]
    #     like_logit = tf.layers.dense(mmoe_output[0], 1, name='like_predictor_mlp')
    #     follow_logit = tf.layers.dense(mmoe_output[1], 1, name='follow_predictor_mlp')
    #     comment_logit = tf.layers.dense(mmoe_output[2], 1, name='comment_predictor_mlp')
    #     forward_logit = tf.layers.dense(mmoe_output[3], 1, name='forward_predictor_mlp')
    #     longview_logit = tf.layers.dense(mmoe_output[4], 1, name='longview_predictor_mlp')

    #     like_logit = tf.reshape(like_logit,[-1, 1])
    #     follow_logit = tf.reshape(follow_logit,[-1, 1])
    #     comment_logit = tf.reshape(comment_logit,[-1, 1])
    #     forward_logit = tf.reshape(forward_logit,[-1, 1])
    #     longview_logit = tf.reshape(longview_logit,[-1, 1])

    #     # pred
    #     like_pred = tf.nn.sigmoid(like_logit) # [-1, 1]
    #     follow_pred = tf.nn.sigmoid(follow_logit)
    #     comment_pred = tf.nn.sigmoid(comment_logit)
    #     forward_pred = tf.nn.sigmoid(forward_logit)
    #     longview_pred = tf.nn.sigmoid(longview_logit)


    #     self.loss_like = tf.losses.log_loss(self.lable_like, like_pred)
    #     self.loss_follow = tf.losses.log_loss(self.lable_follow, follow_pred)
    #     self.loss_comment = tf.losses.log_loss(self.lable_comment, comment_pred)
    #     self.loss_forward = tf.losses.log_loss(self.lable_forward, forward_pred)
    #     self.loss_longview = tf.losses.log_loss(self.lable_longview, longview_pred)

    #     self.loss = self.loss_like + self.loss_follow + self.loss_comment + self.loss_forward + self.loss_longview

        # MF
        self.scores = self.inner_product(self.u_embeddings, self.i_embeddings)
        like_pred = tf.nn.sigmoid(self.scores)
        print("tf.shape(self.lable_like)=", tf.shape(self.lable_like))
        print("1tf.shape(like_pred)=", tf.shape(like_pred))
        like_pred = tf.reshape(like_pred, [-1, 1])
        print("2tf.shape(like_pred)=", tf.shape(like_pred))
        self.loss = tf.losses.log_loss(self.lable_like, like_pred)

        ## optimizer
        if self.optimizer == 'SGD': self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        if self.optimizer == 'RMSProp': self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adam': self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        if self.optimizer == 'Adagrad': self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr)

        ## update parameters
        self.updates = self.opt.minimize(self.loss, var_list=self.var_list)

        ## get top k
        # self.all_ratings = tf.matmul(self.u_embeddings, self.item_embeddings, transpose_a=False, transpose_b=True)
        # self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        # self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices


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

    # query_input =》 [-1, list_size_q=1, dim]     k=v=[-1, list_size_k, dim]
    # retun : [-1, list_size_q=1, nh*dim]
    def set_attention_block(self, query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, mask_flag_k=True):
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


    # inputs = [-1, L, dim]  L=1
    # return = [-1, 1, att_emb_size] ** num_tasks
    def mmoe_layer(self, inputs, att_emb_size=32, num_experts = 1, num_tasks = 1):
        expert_outputs, final_outputs = [], []
        with tf.name_scope('experts_network'):
            for i in range(num_experts):
                expert_layer = tf.layers.dense(inputs, att_emb_size, activation=tf.nn.relu, name='expert{}_'.format(i)+'param')
                expert_outputs.append(tf.expand_dims(expert_layer, axis=3))
        expert_outputs = tf.concat(expert_outputs, 3)  # (batch_size, L, expert_units[-1], num_experts)

        with tf.name_scope('gates_network'):
            for i in range(num_tasks):
                gate_layer = tf.layers.dense(inputs, num_experts, activation=tf.nn.softmax, name='gates{}_'.format(i)+'param')
                expanded_gate_output = tf.expand_dims(gate_layer, 3) # (batch_size, L, num_experts, 1)
                # [-1, L, att, num_experts] * [-1, L, num_experts, 1] = (-1, L, expert_units[-1], 1)
                weighted_expert_output = tf.matmul(expert_outputs, expanded_gate_output) # (batch_size, L, expert_units[-1], 1)
                weighted_expert_output = tf.squeeze(weighted_expert_output, axis=-1)
                final_outputs.append(weighted_expert_output)
        return final_outputs
