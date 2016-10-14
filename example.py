"""
This script is an example of running distributed algorithm. 

The framework is taken from https://www.tensorflow.org/versions/r0.11/how_tos/distributed/index.html.

To launch this application.
"""

import tensorflow as tf
import numpy as np

# flags are arguments passed through command line.
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
  # parsing command line arguments
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Specify the parameter servers and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)
  
  if FLAGS.job_name == "ps":
    print ("\033[1;31mLaunching a parameter server\033[0m\n")
    server.join()
  elif FLAGS.job_name == "worker":
    print ("\033[1;31mLaunching a worker\033[0m\n")
    
    # define computation graph for this worker.
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
        # placeholder have to be "feed" everytime we compute a graph
        X = tf.placeholder("float")
        Y = tf.placeholder("float")
        
        # variables are not cleared between graph computation
        w = tf.Variable(np.random.randn(), name="weight")
        b = tf.Variable(np.random.randn(), name="reminder")

        # defining least square
        cost_op = tf.reduce_mean(tf.square(Y - tf.mul(X, w) - b))
        global_step = tf.Variable(0)
        train_op = tf.train.GradientDescentOptimizer(1/(100+global_step)).minimize(cost_op,  global_step=global_step)
        
        # `saver` class adds ops to save and restore variables to and from *checkpoints*
        saver = tf.train.Saver()
        # Merges all summaries collected in the default graph
        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()

    # Generate data
    training_set_size = 10000
    actual_weight = 2
    actual_bias = 10
    train_X = np.linspace(-2, 2, training_set_size)
    train_Y = actual_weight * train_X + actual_bias + np.random.randn(*train_X.shape) * 0.33

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization, restoring from
    # a checkpoint, and closing when done or an error occurs.
    with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000 steps have completed.
      step = 0
      sess.run(init_op)
      while not sv.should_stop() and step < 1000:          
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        batch_size = 50
        beg = (step * batch_size) % training_set_size
        end = (step * batch_size + batch_size) % training_set_size
        training_batch_X = train_X[beg]
        training_batch_Y = train_Y[end]

        _, step = sess.run([train_op, global_step], feed_dict={X:training_batch_X, Y:training_batch_Y})

        print ("\033[1;31msteps: %4i\033[0m" % step, sess.run([w, b, cost_op], feed_dict={X:train_X, Y:train_Y}))
        # Ask for all the services to stop.
      print ("numerical solution = ", sess.run([w, b], feed_dict={X:train_X, Y:train_Y}))
      print ("actual solution    = ", [actual_weight, actual_bias])
    sv.stop()

    
if __name__ == "__main__":
  tf.app.run()