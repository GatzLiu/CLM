from test_model import *
from utils import *

from params.params_common import MODEL
from models.model_CLM import model_CLM
# from models.model_NCF import model_NCF
# from models.model_NGCF import model_NGCF
# from models.model_LightGCN import model_LightGCN
# from models.model_LGCN import model_LGCN


# [
#           0        1       2      3    4      5         6          7      8     9     10     11     12
#     [ [item_id, time_ms, click, like, follow, comment, forward, longview, pltr, pwtr, pcmtr, pftr, plvtr],  [] .....]
#     []
# ....
# ]

def train_model(para):
    ## paths of data
    train_path = para['DIR'] + 'kuairand_ltr_data_train.json'
    save_model_path = './model_ckpt/model_' + para["MODEL"] + '/clm_model.ckpt'

    ## Load data
    [train_data, item_num] = read_data(train_path)
    print("len(train_data)=",len(train_data), ", item_num=", item_num)

    data = {"item_num": item_num}

    ## define the model
    # if para["MODEL"] == 'CLM': model = model_CLM(data=data, para=para)
    
    # if para["MODEL"] == 'MF': model = model_MF(data=data, para=para)
    # if para["MODEL"] == 'NCF': model = model_NCF(data=data, para=para)
    # if para["MODEL"] == 'NGCF': model = model_NGCF(data=data, para=para)
    # if para["MODEL"] == 'LightGCN': model = model_LightGCN(data=data, para=para)
    # if para["MODEL"] == 'LGCN': model = model_LGCN(data=data, para=para)
    # model = model_MMOE(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # saver
    # saver = tf.train.Saver(max_to_keep = 10)

    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data), para['BATCH_SIZE']))
    batches.append(len(train_data))

    ## training iteratively
    list_auc_epoch = []
    F1_max = 0

    pxtr_bucket_range = np.linspace(0, 1, num=10000)
    for epoch in range(para['N_EPOCH']):
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            real_len_list = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                sample_list, real_len = generate_sample_with_max_len(train_data[sample], para)    # [100, 13]
                sample_list = generate_sample_with_pxtr_bins(train_data[sample], para, pxtr_bucket_range)  # [100, 13+5] 
                train_batch_data.append(sample_list)
                real_len_list.append(real_len)
            
            train_batch_data = np.array(train_batch_data)  # [-1, 100, 13+5],  [pltr_index, pwtr_index, pcmtr_index, plvtr_index, plvtr_index]
            real_len_list = np.array(real_len_list) # [-1]
            # print("train_batch_data[:,:,0]", train_batch_data[:,:,0]) # [-1, 100]
            # print("train_batch_data[:,:,17]", train_batch_data[:,:,17])


            # # 将一个布尔值传递给feed_dict
            # result = sess.run(not_op, feed_dict={model.is_train: True})
