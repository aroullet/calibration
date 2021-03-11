import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
slim = tf.contrib.slim


def temp_scaling(logits_nps, labels_nps, sess, maxiter=50):

    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))

    logits_tensor = tf.constant(logits_nps, name='logits_valid', dtype=tf.float32)
    labels_tensor = tf.constant(labels_nps, name='labels_valid', dtype=tf.int32)

    acc_op = tf.metrics.accuracy(labels_tensor, tf.argmax(logits_tensor, axis=1))

    logits_w_temp = tf.divide(logits_tensor, temp_var)

    # loss
    nll_loss_op = tf.losses.sparse_softmax_cross_entropy(
        labels=labels_tensor, logits=logits_w_temp)
    org_nll_loss_op = tf.identity(nll_loss_op)
    nll_loss_op = tf.cast(nll_loss_op, tf.float32)  # fixes issue with a random int32 showing up in the array

    # optimizer
    optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': maxiter})

    sess.run(temp_var.initializer)
    sess.run(tf.local_variables_initializer())
    org_nll_loss = sess.run(org_nll_loss_op)

    optim.minimize(sess)

    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)
    acc = sess.run(acc_op)

    print("Original NLL: {:.3f}, validation accuracy: {:.3f}%".format(org_nll_loss, acc[0] * 100))
    print("After temperature scaling, NLL: {:.3f}, temperature: {:.3f}, validation accuracy: {:.3f}".format(
        nll_loss, temperature[0], acc[0] * 100))

    return temp_var

with open('arrays1.npy', 'rb') as f:
    labels = np.load(f, allow_pickle=True)
    logits = np.load(f, allow_pickle=True)

t = temp_scaling(logits, labels, sess=tf.Session())
