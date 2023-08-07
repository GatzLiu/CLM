import json
import random as rd
import tensorflow as tf
import numpy as np

def print_params(para):
    for para_name in para:
        print(para_name+':  ',para[para_name])

def print_value(value):
    [inter, loss, f1_max, F1, NDCG] = value
    print('iter: %d loss %.2f f1 %.4f' %(inter, loss, f1_max), end='  ')
    print(F1, NDCG)

def save_embeddings(data, path):
    f = open(path, 'w')
    js = json.dumps(data)
    f.write(js)
    f.write('\n')
    f.close

def read_data1111(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return data, interactions, user_num, item_num


# [
#           0        1       2      3    4      5         6          7      8     9     10     11     12
#     [ [item_id, time_ms, click, like, follow, comment, forward, longview, pltr, pwtr, pcmtr, pftr, plvtr],  [] .....]
#     []
# ....
# ]
def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    row_num = len(data)
    print ("sample number=", row_num, ", data_path=", path)
    print("data[:1]=", data[:1])

    item_num = 0
    for sample in data:
        for item in sample:
            item_num = max(item[0], item_num)
    
    print ("item_num=", item_num) # 25700
    return data, item_num


def cal_auc(sess, epoch_label_like_re, epoch_label_follow_re, epoch_label_comment_re, epoch_label_forward_re, epoch_label_longview_re,
            epoch_like_pred, epoch_follow_pred, epoch_comment_pred, epoch_forward_pred, epoch_longview_pred):
    list_auc = []
    auc_like, auc_op_like = tf.metrics.auc(tf.concat(epoch_label_like_re, 0), tf.concat(epoch_like_pred, 0))
    auc_follow, auc_op_follow = tf.metrics.auc(tf.concat(epoch_label_follow_re, 0), tf.concat(epoch_follow_pred, 0))
    auc_comment, auc_op_comment = tf.metrics.auc(tf.concat(epoch_label_comment_re, 0), tf.concat(epoch_comment_pred, 0))
    auc_forward, auc_op_forward = tf.metrics.auc(tf.concat(epoch_label_forward_re, 0), tf.concat(epoch_forward_pred, 0))
    auc_longview, auc_op_longview = tf.metrics.auc(tf.concat(epoch_label_longview_re, 0), tf.concat(epoch_longview_pred, 0))
    
    sess.run(tf.local_variables_initializer())
    sess.run([auc_op_like, auc_op_follow, auc_op_comment, auc_op_forward, auc_op_longview])
    auc_like_value, auc_follow_value, auc_comment_value, auc_forward_value, auc_longview_value = sess.run(
        [auc_like, auc_follow, auc_comment, auc_forward, auc_longview])
    
    return [auc_like_value, auc_follow_value, auc_comment_value, auc_forward_value, auc_longview_value]
    # auc_like_value = sess.run(auc_like)
    # list_auc_like_value.append(auc_like_value)

# NOTE: add feature [real_length]
def generate_sample(data, para):
    (user, item, time_ms, click, like, follow, comment, forward, longview, user_real_action) = data
    limit_user_real_action = user_real_action
    cur_length = len(limit_user_real_action)
    real_length = 0
    if cur_length >= para['ACTION_LIST_MAX_LEN']:
        limit_user_real_action = limit_user_real_action[-para['ACTION_LIST_MAX_LEN']:]  # tail
        real_length = len(limit_user_real_action)
    else:
        real_length = len(limit_user_real_action)
        list_null_pos = []
        for i in range(para['ACTION_LIST_MAX_LEN'] - cur_length):
            list_null_pos.append(0)
        limit_user_real_action = limit_user_real_action + list_null_pos # first item_id, then 0; use cal attention with mask
    # print("len(limit_user_real_action)=", len(limit_user_real_action), ", real_length=", real_length)
    # print("limit_user_real_action=", limit_user_real_action)
    return [user, item, time_ms, click, like, follow, comment, forward, longview, real_length] + limit_user_real_action


def generate_sample_with_max_len(data, para):
    sample = data
    real_len = len(sample)
    # print ("real_len=", real_len)
    
    # NOTE: real_len <= para['CANDIDATE_ITEM_LIST_LENGTH']
    if real_len == para['CANDIDATE_ITEM_LIST_LENGTH']:
        return sample, real_len
    elif (real_len < para['CANDIDATE_ITEM_LIST_LENGTH']):
        for i in range(para['CANDIDATE_ITEM_LIST_LENGTH'] - real_len):
            sample.append([0,0,0,0,0,0,0,0,0,0,0,0,0])
    return sample, real_len

def generate_sample_with_pxtr_bins(data, para, pxtr_bucket_range):
    sample = []
    for (item_id, time_ms, click, like, follow, comment, forward, longview, pltr, pwtr, pcmtr, pftr, plvtr) in data:
        pltr = max(pltr ,0.00000000)
        pltr = min(pltr ,0.99999999)
        pwtr = max(pwtr ,0.00000000)
        pwtr = min(pwtr ,0.99999999)
        pcmtr = max(pcmtr ,0.00000000)
        pcmtr = min(pcmtr ,0.99999999)
        pftr = max(pftr ,0.00000000)
        pftr = min(pftr ,0.99999999)
        plvtr = max(plvtr ,0.00000000)
        plvtr = min(plvtr ,0.99999999)

        pltr_index = np.searchsorted(pxtr_bucket_range, pltr)
        pwtr_index = np.searchsorted(pxtr_bucket_range, pwtr)
        pcmtr_index = np.searchsorted(pxtr_bucket_range, pcmtr)
        pftr_index = np.searchsorted(pxtr_bucket_range, pftr)
        plvtr_index = np.searchsorted(pxtr_bucket_range, plvtr)

        sample.append([item_id, time_ms, click, like, follow, comment, forward, longview, pltr, pwtr, pcmtr, pftr, plvtr,
                    pltr_index, pwtr_index, pcmtr_index, pftr_index, plvtr_index])
    return sample

def get_order(ranking):
    position_in_ranking = np.argsort(-np.array(ranking))
    order = np.argsort(position_in_ranking) + 1
    return order

def ndcg_for_one_samp(ranking_xtr, ranking_ens, k):
    ranking_xtr = ranking_xtr[:k]
    ranking_ens = ranking_ens[:k]

    order_xtr = get_order(ranking_xtr)
    order_ens = get_order(ranking_ens)
    # print(order_xtr)
    # print(order_ens)
    dcg, idcg = 0, 0
    for i in range(len(ranking_xtr[:k])):
        dcg += ranking_xtr[i] / np.log(order_ens[i] + 1) / np.log(2.0)
        idcg += ranking_xtr[i] / np.log(order_xtr[i] + 1) / np.log(2.0)
    # print(dcg, idcg)
    return dcg / (idcg + 1e-10)
