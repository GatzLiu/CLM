from model_MMOE import *
from test_model import *
from print_save import *
from params import DIR


def train_model(para):
    ## paths of data
    train_path = DIR + 'train_data.json'
    validation_path = DIR + 'validation_data.json'
    # save_embeddings_path = DIR + 'pre_train_embeddings' + str(para['EMB_DIM']) + '.json'

    ## Load data
    [train_data, user_num, item_num] = read_data(train_path)
    print("len(train_data)=",len(train_data), ", user_num=", user_num, ", item_num=", item_num)
    test_data = read_data(validation_path)[0]
    print ("test_data[0:3]=", test_data[0:3])
    # para_test = [train_data, test_data, user_num, item_num, para['TOP_K'], para['TEST_USER_BATCH']]

    data = {'user_num': user_num, "item_num": item_num}

    ## define the model
    model = model_MMOE(data=data, para=para)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ## split the training samples into batches
    batches = list(range(0, len(train_data), para['BATCH_SIZE']))
    batches.append(len(train_data))

    ## training iteratively
    auc_like_value_list = []
    F1_max = 0
    for epoch in range(para['N_EPOCH']):
        auc_like_value_list_epoch = []
        for batch_num in range(len(batches)-1):
            train_batch_data = []
            for sample in range(batches[batch_num], batches[batch_num+1]):
                (user, item, click, like, follow, comment, forward, longview, user_real_action) = train_data[sample]
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
                train_batch_data.append([user, item, click, like, follow, comment, forward, longview, real_length] + limit_user_real_action)

            train_batch_data = np.array(train_batch_data)
            # print ("train_batch_data[:,3].shape=", train_batch_data[:,3].shape)
            # print ("train_batch_data[:,3]=", train_batch_data[:3,3])
            # print("train_batch_data[:,0] = ", train_batch_data[:,0])
            # print(np.shape(train_batch_data[:,0]))
            # print("train_batch_data[:3,9:]=", train_batch_data[:3,9:])

            _, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview, label_like_re, like_pred = sess.run(
                [model.updates, model.loss, model.loss_like, model.loss_follow, model.loss_comment, 
                model.loss_forward, model.loss_longview, model.label_like_re, model.like_pred],
                feed_dict={model.users: train_batch_data[:,0],
                            model.items: train_batch_data[:,1],
                            model.action_list: train_batch_data[:,9:],
                            model.real_length: train_batch_data[:,8],
                            model.label_like: train_batch_data[:,3],
                            model.label_follow: train_batch_data[:,4],
                            model.label_comment: train_batch_data[:,5],
                            model.label_forward: train_batch_data[:,6],
                            model.label_longview: train_batch_data[:,7],
            })

            auc_like, auc_op_like = tf.metrics.auc(tf.reshape(label_like_re, [-1]), tf.reshape(like_pred, [-1]))
            sess.run(tf.local_variables_initializer())
            sess.run(auc_op_like)
            auc_like_value = sess.run(auc_like)
            auc_like_value_list_epoch.append(auc_like_value)
        # print_value([epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview])
        auc_like_value_list.append(auc_like_value_list_epoch)
        print("[epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview] = ", 
        [epoch + 1, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview])
        print ("epoch + 1, auc_like_value(first, tail)=", [epoch + 1, auc_like_value_list_epoch[0], auc_like_value_list_epoch[-1]])
        if not loss < 10 ** 10:
            print ("ERROR, loss big, loss=", loss)
            break
    for row in range(len(auc_like_value_list)):
        print ("epoch+1=", row+1)
        for col in auc_like_value_list[row]:
           print (col, end= " ")
    #     F1, NDCG = test_model(sess, model, para_test)
    #     if F1[1] > F1_max:
    #         F1_max = F1[1]
    #         user_embeddings, item_embeddings = sess.run([model.user_embeddings, model.item_embeddings])
    #     ## print performance
    #     print_value([epoch + 1, loss, F1_max, F1, NDCG])
    #     if not loss < 10 ** 10:
    #         break
    # save_embeddings([user_embeddings.tolist(), item_embeddings.tolist()], save_embeddings_path)
