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
    return data, item_num + 1


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
    dcg, idcg = 0, 0
    for i in range(len(ranking_xtr)):
        dcg += ranking_xtr[i] / np.log(order_ens[i] + 1) / np.log(2.0)
        idcg += ranking_xtr[i] / np.log(order_xtr[i] + 1) / np.log(2.0)
    return dcg / (idcg + 1e-10)

# metrices with @k
def evaluation_F1(order, top_k, positive_item):
    epsilon = 0.1 ** 10
    top_k_items = set(order[0: top_k])
    positive_item = set(positive_item)
    precision = len(top_k_items & positive_item) / max(len(top_k_items), epsilon)
    recall = len(top_k_items & positive_item) / max(len(positive_item), epsilon)
    F1 = 2 * precision * recall / max(precision + recall, epsilon)
    if top_k == 10:
        print(top_k_items)
        print(positive_item)
        print(top_k_items & positive_item)
        print(precision)
        print(recall)
    return F1

def evaluation_NDCG(order, top_k, positive_item):
    top_k_item = order[0: top_k]
    epsilon = 0.1**10
    DCG = 0
    iDCG = 0
    for i in range(top_k):
        if top_k_item[i] in positive_item:
            DCG += 1 / np.log2(i + 2)
    for i in range(min(len(positive_item), top_k)):
        iDCG += 1 / np.log2(i + 2)
    NDCG = DCG / max(iDCG, epsilon)
    return NDCG

def print_pxtr_ndcg(epoch, para, train_data_input, pred_list):
    # ndcg
    k = 100
    list_ltr_ndcg_epoch, list_wtr_ndcg_epoch, list_cmtr_ndcg_epoch, list_ftr_ndcg_epoch, list_lvtr_ndcg_epoch = [], [], [], [], []
    ltr_label_ndcg, wtr_label_ndcg, cmtr_label_ndcg, ftr_label_ndcg, lvtr_label_ndcg = [], [], [], [], []
    click_label_ndcg = []
    for i in range(para['TEST_USER_BATCH']):  #len(pred_list)):
        # pred_list[i]     [max_len]
        # train_data_input[i]->[max_len, 13+5]      train_data_input[i][:,13] # [max_len]
        list_ltr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,13], pred_list[i][:k], k)) # bin
        list_wtr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,14], pred_list[i][:k], k))
        list_cmtr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,15], pred_list[i][:k], k))
        list_ftr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,16], pred_list[i][:k], k))
        list_lvtr_ndcg_epoch.append(ndcg_for_one_samp(train_data_input[i][:k,17], pred_list[i][:k], k))

        click_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,2], pred_list[i][:k], k))
        ltr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,3], pred_list[i][:k], k))
        wtr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,4], pred_list[i][:k], k))
        cmtr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,5], pred_list[i][:k], k))
        ftr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,6], pred_list[i][:k], k))
        lvtr_label_ndcg.append(ndcg_for_one_samp(train_data_input[i][:k,7], pred_list[i][:k], k))

    # ndcg: pxtr-input with pred
    print ("[ep, pxtr ndcg", ", ltr, wtr, cmtr, ftr, lvtr]=", [epoch+1,
            sum(list_ltr_ndcg_epoch)/len(list_ltr_ndcg_epoch),
            sum(list_wtr_ndcg_epoch)/len(list_wtr_ndcg_epoch), sum(list_cmtr_ndcg_epoch)/len(list_cmtr_ndcg_epoch),
            sum(list_ftr_ndcg_epoch)/len(list_ftr_ndcg_epoch), sum(list_lvtr_ndcg_epoch)/len(list_lvtr_ndcg_epoch)])

    # ndcg: pred with action-label
    print ("[ep, label ndcg", ", click, xtr]=", [epoch+1,
        sum(click_label_ndcg)/len(click_label_ndcg), sum(ltr_label_ndcg)/len(ltr_label_ndcg),
        sum(wtr_label_ndcg)/len(wtr_label_ndcg), sum(cmtr_label_ndcg)/len(cmtr_label_ndcg),
        sum(ftr_label_ndcg)/len(ftr_label_ndcg), sum(lvtr_label_ndcg)/len(lvtr_label_ndcg)])

def print_click_ndcg(epoch, para, train_data_input, pred_list):
    f1score = []
    ndcg = []
    for i in range(len(para['TOP_K'])):
        f1score.append([])
        ndcg.append([])
    for i in range(len(para['TOP_K'])):
        for j in range(para['TEST_USER_BATCH']):
            k = para['TOP_K'][i]
            pos_items = np.where(train_data_input[j][:, 2] > 0)[0]
            topk_items = np.argsort(-pred_list[j][:k])
            f1score[i].append(evaluation_F1(topk_items, k, pos_items))
            ndcg[i].append(evaluation_NDCG(topk_items, k, pos_items))
    # ndcg: pred with action-label
    f1score = np.array(f1score)
    ndcg = np.array(ndcg)
    print ("ep", epoch+1, np.mean(f1score, 1), np.mean(ndcg, 1))

