from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np

from optimizer import Optimizer
from input_data import load_data
from model import GCNModel
from preprocessing import preprocess_graph, construct_feed_dict

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 3, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn', 'Model string.')
dim = 2

FLAGS.hidden2 = 2*dim

# Load data
adj = load_data()

print(adj)

# Some preprocessing
adj_norm = preprocess_graph(adj)


# Define placeholders
placeholders = {
    'features': tf.placeholder(tf.float32),
    'adj_norm': tf.placeholder(tf.float32),
    'adj_orig': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = np.identity(adj.shape[0])
num_features = adj.shape[1]

# Create model
model = GCNModel(placeholders, num_features)


norm = 1 / float((adj**2).sum())
print(norm)

# Optimizer

opt = Optimizer(preds=model.reconstructions,labels=tf.reshape(placeholders['adj_norm'], [-1]),norm=norm)


vars = tf.trainable_variables()
#print(vars)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    #preds = sess.run(model.reconstructions, feed_dict=feed_dict)
    # Compute average loss
    avg_cost = outs[1]

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "time=", "{:.5f}".format(time.time() - t))

embed = sess.run(model.embeddings, feed_dict=feed_dict)
print("Optimization Finished!")
embedx,embedy = np.split(embed, 2, axis=1)
print(embedx,embedy)