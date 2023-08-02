from print_save import *
from params import DIR
import tensorflow as tf


# get model
model_path = 'model_ckpt/clm_model.ckpt.meta'
restore_path = 'model_ckpt/clm_model.ckpt'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path)
    saver.restore(sess, restore_path)

print("success")
#     updates = sess.graph.get_operation_by_name('updates')
#     sess.run([updates], feed_dict={

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
