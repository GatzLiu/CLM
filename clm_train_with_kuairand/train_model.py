from test_model import *
from utils import *

from models.model_CLM import model_CLM
from models.model_PRM import model_PRM
from models.model_MLP import model_MLP
from models.model_SUM import model_SUM
from models.model_MUL import model_MUL


# [
#           0        1       2      3    4      5         6          7      8     9     10     11     12
#     [ [item_id, time_ms, click, like, follow, comment, forward, longview, pltr, pwtr, pcmtr, pftr, plvtr],  [] .....]
#     []
# ....
# ]

def train_model(para):
    ## paths of data
    train_path = para['DIR'] + 'kuairand_ltr_data_train.json'
    save_model_path = './model_ckpt/model_' + para["MODEL"] + '/model_' + para["MODEL"] + '.ckpt'

    ## Load data
    [train_data, item_num] = read_data(train_path)
    print("len(train_data)=",len(train_data), ", item_num=", item_num)

    data = {"item_num": item_num}

    ## define the model
    if para["MODEL"] == 'CLM': model = model_CLM(data=data, para=para)
    if para["MODEL"] == 'PRM': model = model_PRM(data=data, para=para)
    if para["MODEL"] == 'MLP': model = model_MLP(data=data, para=para)
    if para["MODEL"] == 'SUM': model = model_SUM(data=data, para=para)
    if para["MODEL"] == 'MUL': model = model_MUL(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # saver
    saver = tf.train.Saver(max_to_keep = 5)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # process data
    train_data_input = []
    real_len_input = []
    real_len_min = 10000
    pxtr_bucket_range = np.linspace(0, 1, num=10000)
    for sample in range(len(train_data)):
        sample_list, real_len = generate_sample_with_max_len(train_data[sample], para)  # [100, 13]
        sample_list = generate_sample_with_pxtr_bins(train_data[sample], para, pxtr_bucket_range)  # [100, 13+5], [pltr_index, pwtr_index, pcmtr_index, plvtr_index, plvtr_index]
        train_data_input.append(sample_list)  
        real_len_input.append(real_len)
        real_len_min = min(real_len_min, real_len)
    train_data_input = np.array(train_data_input)  # [-1, 100, 13+5]
    real_len_input = np.array(real_len_input)
    print ("len(train_data_input)=", len(train_data_input), ", len(real_len_input)=", len(real_len_input))
    print ("real_len_min=", real_len_min)

    ## split the training samples into batches
    batches = list(range(0, len(train_data), para['BATCH_SIZE']))
    batches.append(len(train_data))

    ## training iteratively
    
    F1_max = 0
    for epoch in range(para['N_EPOCH']):
        pred_list = []
        for batch_num in range(len(batches)-1):
            train_batch_data = train_data_input[batches[batch_num]:batches[batch_num+1]]  # [-1, 100, 13+5]
            real_len_batch = real_len_input[batches[batch_num]: batches[batch_num+1]] # [-1]
            # preedict first
            loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias, pred = sess.run(
                [model.loss, model.loss_click, model.loss_sim_order, model.loss_pxtr_reconstruct, model.loss_pxtr_bias,
                model.pred],
                feed_dict={
                    model.item_list: train_batch_data[:,:,0],
                    model.click_label_list: train_batch_data[:,:,2],
                    model.real_length: real_len_batch,
                    model.keep_prob: 1.0,
                    model.like_pxtr_list: train_batch_data[:,:,13],
                    model.follow_pxtr_list: train_batch_data[:,:,14],
                    model.comment_pxtr_list: train_batch_data[:,:,15],
                    model.forward_pxtr_list: train_batch_data[:,:,16],
                    model.longview_pxtr_list: train_batch_data[:,:,17],
                    model.like_pxtr_dense_list: train_batch_data[:,:,8],
                    model.follow_pxtr_dense_list: train_batch_data[:,:,9],
                    model.comment_pxtr_dense_list: train_batch_data[:,:,10],
                    model.forward_pxtr_dense_list: train_batch_data[:,:,11],
                    model.longview_pxtr_dense_list: train_batch_data[:,:,12],
            })
            # then train the model
            _ = sess.run(
                model.updates,
                feed_dict={
                    model.item_list: train_batch_data[:, :, 0],
                    model.click_label_list: train_batch_data[:, :, 2],
                    model.real_length: real_len_batch,
                    model.keep_prob: 0.999,
                    model.like_pxtr_list: train_batch_data[:, :, 13],
                    model.follow_pxtr_list: train_batch_data[:, :, 14],
                    model.comment_pxtr_list: train_batch_data[:, :, 15],
                    model.forward_pxtr_list: train_batch_data[:, :, 16],
                    model.longview_pxtr_list: train_batch_data[:, :, 17],
                    model.like_pxtr_dense_list: train_batch_data[:, :, 8],
                    model.follow_pxtr_dense_list: train_batch_data[:, :, 9],
                    model.comment_pxtr_dense_list: train_batch_data[:, :, 10],
                    model.forward_pxtr_dense_list: train_batch_data[:, :, 11],
                    model.longview_pxtr_dense_list: train_batch_data[:, :, 12],
                })
            pred_list.append(pred) # pred = [-1, max_len]

        if ((epoch+1) == 5) or ((epoch+1) == 10):
            print ("start save model , epoch+1=", epoch+1)
            save_path = saver.save(sess, save_model_path, global_step=epoch+1)
        #                            1              2              0.1                   1
        print ("[epoch+1, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias] = ", 
                [epoch+1, loss, loss_click, loss_sim_order, loss_pxtr_reconstruct, loss_pxtr_bias])
        
        pred_list = np.concatenate(pred_list, axis=0) # pred_list = [-1, max_len]
        # print ("len(pred_list)=", len(pred_list), ", len(train_batch_data)=", len(train_data_input))

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
        
        if not loss < 10 ** 10:
            print ("ERROR, loss big, loss=", loss)
            break


            # # 将一个布尔值传递给feed_dict
            # result = sess.run(not_op, feed_dict={model.is_train: True})
