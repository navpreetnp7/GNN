import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS


class Optimizer(object):
    def __init__(self, preds, labels, norm):
        preds_sub = preds #* (1/norm)
        labels_sub = labels #* norm

        #self.cost =  tf.reduce_mean(tf.losses.mean_squared_error(predictions=preds_sub, labels=labels_sub))
        #self.cost =  tf.reduce_mean(tf.square(labels_sub - preds_sub))
        self.cost =   tf.reduce_sum(tf.square(labels_sub - preds_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
