
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn

def linear_set_attention_block(query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, m_size=32, iter_num=0):
    ## poly encoder
    with tf.name_scope(name):
        I = tf.get_variable(name + "_i_trans_matrix",(1, m_size, col), initializer=tf.truncated_normal_initializer(stddev=5.0)) # [-1, m_size, col]
        I = tf.tile(I, [tf.shape(query_input)[0],1,1])
        H = set_attention_block(I, action_list_input, name + "_ele2clus", mask, col, 1, action_item_size, att_emb_size, True, True)    #[-1, m_size, nh*dim]
        H_list = [H]
        for l in range(iter_num):
            H += set_attention_block(H, H, name + "_sa_clus2clus_{}".format(l), mask, att_emb_size, 1, att_emb_size, att_emb_size, False, False)
            H = CommonLayerNorm(H, scope='ln1_clus2clus_{}'.format(l))
            H += tf.layers.dense(tf.nn.relu(H), att_emb_size, name='ffn_clus2clus_{}'.format(l))
            H = CommonLayerNorm(H, scope='ln2_clus2clus_{}'.format(l))
            H_list.append(H)
        H = tf.reduce_sum(H_list, axis=0)
        res = set_attention_block(query_input, H, name + "_clus2ele", mask, col, nh, att_emb_size, att_emb_size, True, False)
    return res

# query_input =》 [-1, list_size_q=1, dim]     k=v=[-1, list_size_k, dim]
# retun : [-1, list_size_q=1, nh*dim]
def set_attention_block(query_input, action_list_input, name, mask, col, nh=8, action_item_size=152, att_emb_size=64, if_mask=True, mask_flag_k=True):
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


def CommonLayerNorm(inputs,
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


def add_position_emb(query_input, pxtr_dense, seq_length, pxtr_num, dim, decay, name):
    with tf.name_scope(name):
        pos_embeddings = tf.get_variable(name+"_pos_embeddings", (seq_length, dim), initializer=tf.truncated_normal_initializer(stddev=5.0))  # [-1, m_size, col]
        position_in_ranking = tf.contrib.framework.argsort(tf.stop_gradient(pxtr_dense), axis=1)
        order = tf.contrib.framework.argsort(position_in_ranking, axis=1)
        pos_emb = tf.gather(pos_embeddings, order)
        return decay * tf.reshape(pos_emb, [-1, seq_length, pxtr_num * dim]) + query_input
        # return tf.concat([tf.reshape(pos_emb, [-1, seq_length, pxtr_num * dim]), query_input], -1)

def pxtr_transformer(pxtr_input, listwise_len, pxtr_num, dim, name):
    pxtr_input = tf.reshape(pxtr_input, [-1, pxtr_num, dim])
    pxtr_input = set_attention_block(pxtr_input, pxtr_input, name + "_pxtr_transformer", 0, dim, 1, dim, dim, False, False)
    return tf.reshape(pxtr_input, [-1, listwise_len, pxtr_num * dim])

def sigmoid(x):
    return 2 * tf.nn.sigmoid(x) - 1

def sim_order_reg_core(seq_1, seq_2, if_norm, length):
    seq_conc = tf.concat([tf.expand_dims(seq_1, -1), tf.expand_dims(seq_2, -1)], -1)
    seq_cut = seq_conc[:, 0: length, :]
    random_index = tf.random.uniform((1, 200), minval=0, maxval=tf.cast(length, dtype=tf.float32))
    random_index = tf.squeeze(tf.cast(tf.floor(random_index), dtype=tf.int64))
    seq_samp = tf.gather(seq_cut, random_index, axis=1)
    seq = tf.reshape(seq_samp, [-1, 2])
    if if_norm:
        seq_mean, seq_var = tf.nn.moments(tf.stop_gradient(seq), axes=0)
        seq_norm = (seq - tf.expand_dims(seq_mean, 0)) / (tf.sqrt(tf.expand_dims(seq_var, 0)) + 0.1 ** 10)
    else:
        seq_norm = seq
    seq_resh = tf.reshape(seq_norm, [-1, 2, 2])
    # attention! sigmoid(x) = 2 * tf.nn.sigmoid(x) - 1
    # TODO: replace with tanh(x)
    reg_loss = tf.multiply(sigmoid(seq_resh[:, 0, 0] - seq_resh[:, 1, 0]), sigmoid(seq_resh[:, 0, 1] - seq_resh[:, 1, 1]))
    return -tf.reduce_mean(reg_loss)

def sim_order_reg(pred, pxtr, weight, length):
    reg_loss = 0
    for i, w in enumerate(weight):
        reg_loss += w * sim_order_reg_core(pred, pxtr[:, :, i], True, length)
        # reg_loss += weaken_bad_pxtr_weight * w * sim_order_reg_core(pred, -1 / (pxtr[:, :, i] + 0.1 ** 10), True, length)
    return reg_loss