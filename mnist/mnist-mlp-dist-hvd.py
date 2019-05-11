# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = tf.app.flags.FLAGS

def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    #net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net

def main(_):
    # load mnist dataset
    mnist = read_data_sets("/tmp/data", one_hot=True)

    # the model
    images = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.int32, [None, 10])

    logits = model(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=2000)]

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

    train_op = optimizer.minimize(loss, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N)

    with tf.Session() as sess:
        sess.run(global_step)
        for i in range(30000):
            img_batch, label_batch = mnist.train.next_batch(32)
            _, restul_loss, result_step = sess.run(
                [train_op, loss, global_step],
                feed_dict={images: img_batch, labels: label_batch})
            if result_step % 100 == 0:
                print("Train step %d, loss: %f" % (result_step, result_loss))

if __name__ == "__main__":
    tf.app.run()
