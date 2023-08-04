from test_model import *
from utils import *

from params.params_common import MODEL
from models.model_CLM import model_CLM
# from models.model_NCF import model_NCF
# from models.model_NGCF import model_NGCF
# from models.model_LightGCN import model_LightGCN
# from models.model_LGCN import model_LGCN

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

    sum = 0
    count_100 = 0
    count_non_100 = 0

    a_count_100 = 0
    a_count_non_100 = 0
    for epoch in range(para['N_EPOCH']):
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                sum = sum+1
                # test
                if len(train_data[sample]) == 100 :
                    count_100 = count_100 + 1
                elif len(train_data[sample]) < 100:
                    count_non_100 = count_non_100 + 1
                sample_list = generate_sample_v2(train_data[sample], para)
                
                if len(sample_list) == 100 :
                    a_count_100 = a_count_100 + 1
                elif len(sample_list) < 100:
                    a_count_non_100 = a_count_non_100 + 1
                # train_batch_data.append(sample_list)
    
    print ("sum =",  sum)
    print ("count_100=", count_100, ", count_non_100=", count_non_100)
    print ("a_count_100=", a_count_100, ", a_count_non_100=", a_count_non_100)
