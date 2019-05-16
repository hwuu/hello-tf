# -*- coding: utf-8 -*-

# Ref 1: 分布式 TensorFlow 入门教程 -- https://zhuanlan.zhihu.com/p/35083779
# Ref 2: 分布式 Tensorflow 部署和开发 -- https://blog.csdn.net/luodongri/article/details/52596780
# Ref 3: 分布式 Tensorflow 代码固定模式 -- http://www.voidcn.com/article/p-eojgqsmv-ys.html
# Ref 4: Tensorflow 常见错误 -- http://whatbeg.com/2018/12/05/tensorflowtips.html
# Ref 5: https://github.com/yahoo/TensorFlowOnSpark/issues/244#issuecomment-372503145
#
# Start:
#     python mnist-mlp-dist-ps.py --tf-config={\"cluster\":{\"ps\":[\"localhost:2222\",\"localhost:2223\"],\"worker\":[\"localhost:2224\",\"localhost:2225\"]},\"task\":{\"type\":\"ps\",\"index\":0}}
#     python mnist-mlp-dist-ps.py --tf-config={\"cluster\":{\"ps\":[\"localhost:2222\",\"localhost:2223\"],\"worker\":[\"localhost:2224\",\"localhost:2225\"]},\"task\":{\"type\":\"ps\",\"index\":1}}
#     python mnist-mlp-dist-ps.py --tf-config={\"cluster\":{\"ps\":[\"localhost:2222\",\"localhost:2223\"],\"worker\":[\"localhost:2224\",\"localhost:2225\"]},\"task\":{\"type\":\"worker\",\"index\":0}}
#     python mnist-mlp-dist-ps.py --tf-config={\"cluster\":{\"ps\":[\"localhost:2222\",\"localhost:2223\"],\"worker\":[\"localhost:2224\",\"localhost:2225\"]},\"task\":{\"type\":\"worker\",\"index\":1}}
#
# Note:
#  - In AML, if we set the total # of steps to be small (2000 or so), in most of the cases worker_1 will indefinitely
#    print the error message '2019-05-15 08:32:54.391979: I tensorflow/core/distributed_runtime/master.cc:224] CreateSession still waiting for response from worker: /job:worker/replica:0/task:0'.
#  - The solution is to let each worker run for a longer time (for example, by setting # of steps larger).
#  - My guess to why this happens (see ref 5). This may be what should happen ideally:
#                    t0           t1              t2           t3
#        worker_0    initialize   start master    start task   exit
#        worker_1    initialize   start master    start task   exit
#
#    However, if worker_0 runs sufficiently fast, the sequence looks like this:
#                    t0           t1              t2           t3     t4
#        worker_0    initialize   start master    start task   exit
#        worker_1    initialize   start master                        start task (should fail)
#
#    This may be due to some implementation detail of MonitoredTrainingSession. But I'm not sure.
#  - Conclusion: It seems better to use Supervisor other than MonitoredTrainingSession.
#

import os
import json
import argparse
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# Parameters from command line.
args_ = None

def model(images):
    """Define a simple mnist classifier"""
    net = tf.layers.dense(images, 500, activation=tf.nn.relu)
    #net = tf.layers.dense(net, 500, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=None)
    return net

def main(_):
    tf_config_str = args_.tf_config
    if tf_config_str is None:
        tf_config_str = os.environ.get("TF_CONFIG")
    print("TF_CONFIG: %s" % tf_config_str)
    tf_config_dict = json.loads(tf_config_str)
    cluster_spec_dict = tf_config_dict.get('cluster')
    job_name = tf_config_dict.get('task', {}).get('type')
    task_index = tf_config_dict.get('task', {}).get('index')
    print("Job name: " + job_name)
    print("Task index: " + str(task_index))
    
    import socket
    print(socket.gethostbyname(socket.gethostname()))
    
    # create the cluster
    cluster = tf.train.ClusterSpec(cluster_spec_dict)

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
            ps_device="/job:ps/cpu:0",
            worker_device="/job:worker/task:%d/cpu:0" % (task_index),
            cluster=cluster)):
            # load mnist dataset
            mnist = read_data_sets(args_.data_dir, one_hot=True)

            # the model
            images = tf.placeholder(tf.float32, [None, 784])
            labels = tf.placeholder(tf.int32, [None, 10])

            logits = model(images)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=20000)]

            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

            if args_.in_sync_mode:
                # sychronous training
                # use tf.train.SyncReplicasOptimizer wrap optimizer
                # ref: https://www.tensorflow.org/api_docs/python/tf/train/SyncReplicasOptimizer
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=cluster.num_tasks(job_name),
                    total_num_replicas=cluster.num_tasks(job_name))
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
                #checkpoint_dir="/tmp/checkpoints",
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
    parser = argparse.ArgumentParser(description='Distributed TF with parameter servers')
    parser.add_argument("--data-dir", type=str, default="/tmp/data",
        help='Data folder. MNIST data files will be downloaded if they do not exist.')
    parser.add_argument("--tf-config", type=str, default=None,
        help='TF config in json format.')
    parser.add_argument("--in-sync-mode", action="store_true",
        help='Whether distributed TF should run in synchronized mode.')
    args_ = parser.parse_args()
    tf.app.run()
