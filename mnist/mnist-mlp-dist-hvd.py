# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

FLAGS = tf.app.flags.FLAGS

#

def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    #net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net

#

def train(dist=True):
    # load mnist dataset
    mnist_dataset = read_data_sets("/tmp/data", one_hot=True)

    # the model
    images = tf.placeholder(tf.float32, [None, 784])
    labels = tf.placeholder(tf.int32, [None, 10])

    logits = model(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=20000)]

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

    config = tf.ConfigProto()
    checkpoint_dir = "/tmp/train_logs"

    if dist:
        import horovod.tensorflow as hvd
        # Initialize Horovod
        hvd.init()
        # Pin GPU to be used to process local rank (one GPU per process)
        #config.gpu_options.visible_device_list = str(hvd.local_rank())
        checkpoint_dir = "/tmp/train_logs" if hvd.rank() == 0 else None
        # Add hook to broadcast variables from rank 0 to all other processes during
        # initialization.
        hooks.append(hvd.BroadcastGlobalVariablesHook(0))
        # Wrap the original optimizer with Horovod.
        optimizer = hvd.DistributedOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=global_step, aggregation_method=tf.AggregationMethod.ADD_N)

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        config=config,
        hooks=hooks) as mon_sess:
        while not mon_sess.should_stop():
            img_batch, label_batch = mnist_dataset.train.next_batch(32)
            # Perform synchronous training.
            _, loss_val, step_val = mon_sess.run(
                [train_op, loss, global_step],
                feed_dict={images: img_batch, labels: label_batch})
            if step_val % 1000 == 0:
                print("Train step %d, loss: %f" % (step_val, loss_val))

#

def main(_):
    train(dist=False)

#

if __name__ == "__main__":
    tf.app.run()
