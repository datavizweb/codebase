#!/usr/bin/python

##
## DNN Classifier using Tensorflow
##
import tensorflow.python.platform

import pdb
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


## control parameters
class Params:
    """
    parameters 
    """
    num_classes = 2
    feat_dim = 2
    hidden_dim = 5
    batch_size = 100 
    learning_rate = 0.1
    num_epochs = 100
    verbose = False


class DataSet(object):
    """
    generate moon data using sklearn
    """
    def __init__(self, dtype="moon", num_classes=2, batch_size=100, dsplit=0.8):
        """
        initialize the batch size
        """
        self.batch_size = batch_size
        self.dtype = dtype
        self.num_classes = num_classes
        self.dsplit = dsplit


    def train_data(self, num_data=2000, stddev=0.10):
        """
        generate the moon/linear data
        """
        if self.dtype == "moon":
            feat_vec, labels = datasets.make_moons(num_data, noise=stddev)
        elif self.dtype == "linear":
            feat_vec, labels = make_blobs(n_samples=num_data, n_features=2, 
                                          centers=2, cluster_std=1.7)
        else:
            feat_vec, labels = datasets.make_moons(num_data, noise=stddev)

        ##
        ## we need to have these in numpy matrix format
        ##
        feats_vecs = np.matrix(feat_vec).astype(np.float32)
        labels = np.array(labels).astype(dtype=np.uint8)

        # Convert the int numpy array into a one-hot matrix.
        labels_onehot = (np.arange(self.num_classes) == labels[:, None]).astype(np.float32)

        ##
        ## create train and test set
        ##
        train_set_size = int(self.dsplit * num_data)

        self.feats_vecs = feats_vecs[:train_set_size,:]
        self.tfeats_vecs = feats_vecs[train_set_size:,:] 
        self.labels_onehot = labels_onehot[:train_set_size]
        self.tlabels_onehot = labels_onehot[train_set_size:]

        # Return a pair of the feature matrix and the one-hot label matrix.
        return self.feats_vecs, self.labels_onehot


    def next(self):
        """
        get the next batch data
        """
        if self.current >= self.size:
            raise StopIteration

        start = self.current
        end = start + self.batch_size

        fv = self.feats_vecs[start :  end, :]
        labels = self.labels_onehot[start : end, :]
        self.current = end

        return fv, labels


    def __iter__(self):
        """
        return and object to itself
        """
        self.current = 0
        self.size = self.feats_vecs.shape[0]

        return self

  
    def test_data(self):
        """
        return test data
        """
        # Return a pair of the feature matrix and the one-hot label matrix.
        return self.tfeats_vecs, self.tlabels_onehot


    def plot(self):
        """
        plot train and test
        """
        xdata = self.feats_vecs[:, 0]
        ydata = self.feats_vecs[:, 1]

        plt.plot(xdata, ydata, marker='o', linestyle="None")
        plt.hold(True)

        xdata = self.tfeats_vecs[:, 0]
        ydata = self.tfeats_vecs[:, 1]

        plt.plot(xdata, ydata, 'rs', linestyle="None")
        plt.show()


class NNModel:
    """
    neural network model
    """
    def __init__(self, nonlinear="relu"):
        if nonlinear == "relu":
            self.nlf = tf.nn.relu
        elif nonlinear == "tanh":
            self.nlf = tf.nn.tanh
        elif nonlinear == "sigmoid":
            self.nlf = tf.nn.sigmoid
        else:
            self.nlf = tf.nn.relu

    def network(self, x, weights, bias):
        """
        """
        ## layer 1
        W1, b1 = weights["w1"], bias["b1"]
        l1 = self.nlf(tf.matmul(x, W1) + b1, name="layer1")

        ## layer2
        W2, b2 = weights["w2"], bias["b2"]
        l2 = self.nlf(tf.matmul(l1, W2) + b2, name="layer2")

        ## layer3
        W3, b3 = weights["w3"], bias["b3"]
        l3 = tf.add(tf.matmul(l2, W3), b3, name="lastlayer")

        return l3


    def network_linear(self, x, weights, bias):
        """
        """
        ## layer 1
        W1, b1 = weights["w1"], bias["b1"]

        return tf.add(tf.matmul(x, W1), b1, name="layer1")


    def init_weights(self, num_feats, hidden_dim, num_classes,
                     init_type = "rnd_normal"):
        """
        initialize the weights
        """
        if init_type == "rnd_normal":
            initv = tf.random_normal
        elif init_type == "zeros":
            initv = tf.zeros
        else:
            initv = tf.zeros

        weights = {
            "w1" : tf.Variable(initv([num_feats, hidden_dim]), name="weight1"),
            "w2" : tf.Variable(initv([hidden_dim, hidden_dim]), name="weight2"),
            "w3" : tf.Variable(initv([hidden_dim, num_classes]), name="weight3")
        }

        bias = {
            "b1" : tf.Variable(initv([1, hidden_dim]), name="bias1"),
            "b2" : tf.Variable(initv([1, hidden_dim]), name="bias2"),
            "b3" : tf.Variable(initv([1, num_classes]), name="bias3")
        }

        return weights, bias


    def init_weights_linear(self, num_feats, hidden_dim, num_classes,
                     init_type = "rnd_normal"):
        """
        initialize the weights
        """
        if init_type == "rnd_normal":
            initv = tf.random_normal
        elif init_type == "zeros":
            initv = tf.zeros
        else:
            initv = tf.zeros

        weights = {
            "w1" : tf.Variable(initv([num_feats, num_classes]), name="weight1"),
        }

        bias = {
            "b1" : tf.Variable(initv([1, num_classes]), name="bias1"),
        }

        return weights, bias


def train_model():
    """
    train the model
    """
    p = Params()
    data = DataSet(dtype="moon")

    ##
    ## get the train and test
    ##
    train_data, train_labels = data.train_data()
    test_data, test_labels = data.test_data()

    # Get the shape of the training data.
    train_size, feat_dim = train_data.shape

    ##
    ## creation of computation graph
    ##
    x = tf.placeholder(tf.float32, shape = [None, feat_dim])
    y_ = tf.placeholder(tf.float32, shape = [None, p.num_classes])
    
    ##
    ## create single layer nn model
    ##
    model = NNModel()
    weights, bias = model.init_weights(feat_dim, 
                                       p.hidden_dim, p.num_classes)

    ## final softmax layer
    with tf.name_scope("Wx_b") as scope:
        y = tf.nn.softmax(model.network(x, weights, bias), name="softmax")

    ##
    ## collect stat
    ##
    y_hist = tf.histogram_summary("y", y)

    ##
    ## define the cost function and optimization to use
    ##
    cross_entropy = -1 * tf.reduce_sum(y_ * tf.log(y))

    ## cost = tf.train.GradientDescentOptimizer(p.learning_rate).minimize(cross_entropy)
    cost = tf.train.AdagradOptimizer(p.learning_rate).minimize(cross_entropy)

    ##
    ## Evaluation.
    ##
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()

    ##
    ## now create the session and run the train
    ##
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        train_writer = tf.train.SummaryWriter("./train_logs", sess.graph)
        test_writer = tf.train.SummaryWriter("./test_logs", sess.graph)

        ##
        ## Run the model training for num_epochsIterate and train.
        ##
        for epoch in xrange(p.num_epochs):
            if (epoch % 10) == 0:
                summary, acc = sess.run([merged, accuracy], 
                                        feed_dict = {x: test_data, 
                                                     y_: test_labels})
                test_writer.add_summary(summary, epoch)
                print "Accuracy at step %d %f" % (epoch, acc)

            for x_data, y_labels in data:
                train_summary, _ = sess.run([merged, cost], 
                                            feed_dict = {x: x_data, y_: y_labels})

            if (epoch % 10) == 0:
                train_writer.add_summary(train_summary, epoch)

        print "Accuracy:", sess.run(accuracy, 
                                    feed_dict = {x: test_data, 
                                                 y_: test_labels})

    ##
    ## plot the data
    ##
    data.plot()

if __name__ == '__main__':
    train_model()
    

