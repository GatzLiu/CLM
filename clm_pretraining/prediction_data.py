from print_save import *
from params import DIR
import tensorflow as tf



pred_data_path = DIR + 'train_data_pred.json'
model_path = 'model_ckpt/clm_model.ckpt.meta'
restore_path = 'model_ckpt/clm_model.ckpt'

## Load data
pred_data, _, _ = read_data(pred_data_path)
print ("pred_data[0:3]=", pred_data[0:3])

## split the pred-samples into batches
batches = list(range(0, len(pred_data), para['BATCH_SIZE']))
batches.append(len(pred_data))

for epoch in range(para['N_EPOCH']):
    for batch_num in range(len(batches)-1):
        pred_batch_data = []
        for sample in range(batches[batch_num], batches[batch_num+1]):
            (user, item, click, like, follow, comment, forward, longview, user_real_action) = pred_data[sample]
            limit_user_real_action = user_real_action


with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, restore_path)

    user = sess.graph.get_tensor_by_name('users:0')
    item = sess.graph.get_tensor_by_name('items:0')
    action_list = sess.graph.get_tensor_by_name('action_list:0')
    label_like = sess.graph.get_tensor_by_name('label_like:0')
    label_follow = sess.graph.get_tensor_by_name('label_follow:0')
    label_comment = sess.graph.get_tensor_by_name('label_comment:0')
    label_forward = sess.graph.get_tensor_by_name('label_forward:0')
    label_longview = sess.graph.get_tensor_by_name('label_longview:0')

    feed_dict = {
        user:,
        item:,
        action_list:,
        label_like:,
        label_follow:,
        label_comment:,
        label_forward:,
        label_longview:,
    }
    # loss  sess.graph.get_tensor_by_name('')
    loss_like = sess.graph.get_tensor_by_name('log_loss/value:0')
    loss_follow = sess.graph.get_tensor_by_name('log_loss_1/value:0')
    loss_comment = sess.graph.get_tensor_by_name('log_loss_2/value:0')
    loss_forward = sess.graph.get_tensor_by_name('log_loss_3/value:0')
    loss_longview = sess.graph.get_tensor_by_name('log_loss_4/value:0')
    loss = sess.graph.get_tensor_by_name('add_3:0')

    # label: cal auc
    label_like_re = sess.graph.get_tensor_by_name('label_like_re:0')
    label_follow_re = sess.graph.get_tensor_by_name('label_follow_re:0')
    label_comment_re = sess.graph.get_tensor_by_name('label_comment_re:0')
    label_forward_re = sess.graph.get_tensor_by_name('label_forward_re:0')
    label_longview_re = sess.graph.get_tensor_by_name('label_longview_re:0')

    # pred: cal auc & save
    like_pred = sess.graph.get_tensor_by_name('like_pred:0')
    follow_pred = sess.graph.get_tensor_by_name('follow_pred:0')
    comment_pred = sess.graph.get_tensor_by_name('comment_pred:0')
    forward_pred = sess.graph.get_tensor_by_name('forward_pred:0')
    longview_pred = sess.graph.get_tensor_by_name('longview_pred:0')

    # opt
    updates = sess.graph.get_operation_by_name('GradientDescent/GradientDescent/-apply')

    # for   batch

    _, model_loss, model_loss_like, model_loss_follow, model_loss_comment, model_loss_forward, model_loss_longview, \
    model_label_like_re, model_label_follow_re, model_label_comment_re, model_label_forward_re, model_label_longview_re, \
    model_like_pred, model_follow_pred, model_comment_pred, model_forward_pred, model_longview_pred = \
    sess.run(
        [updates, loss, loss_like, loss_follow, loss_comment, loss_forward, loss_longview,
            label_like_re, label_follow_re, label_comment_re, label_forward_re, label_longview_re,
            like_pred, follow_pred, comment_pred, forward_pred, longview_pred], 
        feed_dict=feed_dict
    )

    print ("model_like_pred=", model_like_pred)

print("success")

#     })


# feed_dict={model.users: train_batch_data[:,0],
#                             model.items: train_batch_data[:,1],
#                             model.action_list: train_batch_data[:,9:],
#                             model.real_length: train_batch_data[:,8],
#                             model.label_like: train_batch_data[:,3],
#                             model.label_follow: train_batch_data[:,4],
#                             model.label_comment: train_batch_data[:,5],
#                             model.label_forward: train_batch_data[:,6],
#                             model.label_longview: train_batch_data[:,7],


# #
# output = sess.graph.get_tensor_by_name('output:0')
# accuracy = sess.graph.get_tensor_by_name('accuracy:0')
# train_step = sess.graph.get_operation_by_name('train')


# 3、预测
# print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))


# # save pred    dataset/KuaiRand/DataWithPred

# # jupyter 拼接

if __name__ == '__main__':
    mmoe_model_pred()