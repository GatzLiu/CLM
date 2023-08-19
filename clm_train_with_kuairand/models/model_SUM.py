## basic baseline MF_BPR

import tensorflow as tf

class model_SUM(object):
    def __init__(self, data, para):
        ## model hyper-params
        self.model_name = 'SUM'
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

        # 2 reshape
        self.click_label_list_re = tf.reshape(self.click_label_list, [-1, self.max_len])
        self.real_length_re = tf.reshape(self.real_length, [-1, 1])
        #   pxtr dense
        self.pltr_dense_list = tf.reshape(self.like_pxtr_dense_list, [-1, self.max_len])
        self.pwtr_dense_list = tf.reshape(self.follow_pxtr_dense_list, [-1, self.max_len])
        self.pcmtr_dense_list = tf.reshape(self.comment_pxtr_dense_list, [-1, self.max_len])
        self.pftr_dense_list = tf.reshape(self.forward_pxtr_dense_list, [-1, self.max_len])
        self.plvtr_dense_list = tf.reshape(self.longview_pxtr_dense_list, [-1, self.max_len])

        logits = para['alpha_ltr'] * self.pltr_dense_list
        logits += para['alpha_wtr'] * self.pwtr_dense_list
        logits += para['alpha_cmtr'] * self.pcmtr_dense_list
        logits += para['alpha_ftr'] * self.pftr_dense_list
        logits += para['alpha_lvtr'] * self.plvtr_dense_list
        self.pred = tf.nn.sigmoid(logits)

        #   5.5 loss
        self.loss = tf.constant(0)
        self.updates = tf.constant(0)