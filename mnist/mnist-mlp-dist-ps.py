# -*- coding: utf-8 -*-

# Ref: https://zhuanlan.zhihu.com/p/35083779
#
# Start ps instances:
#    - python mnist-mlp-dist-ps.py --tf_config={\"cluster\":{\"ps\":\"localhost:2222,localhost:2223\",\"worker\":\"localhost:2224,localhost:2225\"},\"task\":{\"type\":\"ps\",\"index\":0}}
#    - python mnist-mlp-dist-ps.py --tf_config={\"cluster\":{\"ps\":\"localhost:2222,localhost:2223\",\"worker\":\"localhost:2224,localhost:2225\"},\"task\":{\"type\":\"ps\",\"index\":1}}
#
# Start worker instances:
#    - python mnist-mlp-dist-ps.py --tf_config={\"cluster\":{\"ps\":\"localhost:2222,localhost:2223\",\"worker\":\"localhost:2224,localhost:2225\"},\"task\":{\"type\":\"worker\",\"index\":0}}
#    - python mnist-mlp-dist-ps.py --tf_config={\"cluster\":{\"ps\":\"localhost:2222,localhost:2223\",\"worker\":\"localhost:2224,localhost:2225\"},\"task\":{\"type\":\"worker\",\"index\":1}}
#

import os
import json
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.app.flags.DEFINE_string("tf_config", "", "config in json format")
tf.app.flags.DEFINE_boolean("is_sync", False, "using synchronous training or not")

FLAGS = tf.app.flags.FLAGS

def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    #net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net

def main(_):
    tf_config = FLAGS.tf_config.strip()
    if tf_config == "":
        tf_config = os.environ.get("TF_CONFIG")
    print(tf_config)
    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get('cluster')
    job_name = tf_config_json.get('task', {}).get('type')
    task_index = tf_config_json.get('task', {}).get('index')

    # create the cluster configured by `ps_hosts' and 'worker_hosts'
    cluster = tf.train.ClusterSpec(cluster)

    # create a server for local task
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == "ps":
        server.join()  # ps hosts only join
    elif job_name == "worker":
        # workers perform the operation
        # ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(FLAGS.num_ps)

        # Note: tf.train.replica_device_setter automatically place the paramters (Variables)
        # on the ps hosts (default placement strategy:  round-robin over all ps hosts, and also
        # place multi copies of operations to each worker host
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % (task_index),
            cluster=cluster)):
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

            if FLAGS.is_sync:
                # asynchronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=len(cluster["worker"]),
                    total_num_replicas=len(cluster["worker"]))
                # create the hook which handles initialization and queues
                hooks.append(optimizer.make_session_run_hook(task_index==0))

            train_op = optimizer.minimize(
                loss,
                global_step=global_step,
                aggregation_method=tf.AggregationMethod.ADD_N)

            # The MonitoredTrainingSession takes care of session initialization,
            # restoring from a checkpoint, saving to a checkpoint, and closing when done
            # or an error occurs.
            with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(task_index == 0),
                checkpoint_dir="/tmp/checkpoints",
                hooks=hooks) as mon_sess:
                while not mon_sess.should_stop():
                    # mon_sess.run handles AbortedError in case of preempted PS.
                    img_batch, label_batch = mnist.train.next_batch(100)
                    _, ls, step = mon_sess.run(
                        [train_op, loss, global_step],
                        feed_dict={images: img_batch, labels: label_batch})
                    if step % 100 == 0:
                        print("Train step %d, loss: %f" % (step, ls))

if __name__ == "__main__":
    tf.app.run()