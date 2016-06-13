#!/usr/bin/python

import tensorflow.python.platform

import pdb
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt


## control parameters
class Params:
    """
    parameters 
    """
    num_classes = 2
    feat_dim = 2
    hidden_dim = 5
    batch_size = 50 
    learning_rate = 0.01
    num_epochs = 100
    verbose = False


class DataSet(object):
    """
    generate moon data using sklearn
    """
    def __init__(self, dtype="moon", num_classes=2, batch_size=100):
        """
        initialize the batch size
        """
        self.batch_size = 100
        self.dtype = dtype
        self.num_classes = num_classes

    def train_data(self, num_data=2000, stddev=0.10):
        """
        generate the moon data
        """
        feat_vec, labels = datasets.make_moons(2000, noise=stddev)
        self.stddev = stddev

        ##
        ## we need to have these in numpy matrix format
        ##
        self.feats_vecs = np.matrix(feat_vec).astype(np.float32)
        labels = np.array(labels).astype(dtype=np.uint8)

        # Convert the int numpy array into a one-hot matrix.
        self.labels_onehot = (np.arange(self.num_classes) == labels[:, None]).astype(np.float32)

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

  
    def test_data(self, num_data=100):
        """
        """
        feat_vec, labels = datasets.make_moons(num_data, noise=self.stddev)

        ##
        ## we need to have these in numpy matrix format
        ##
        self.tfeats_vecs = np.matrix(feat_vec).astype(np.float32)
        labels = np.array(labels).astype(dtype=np.uint8)
        self.tlabels = (np.arange(self.num_classes) == labels[:, None]).astype(np.float32)

        # Return a pair of the feature matrix and the one-hot label matrix.
        return self.tfeats_vecs, self.tlabels


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

    def init_weights(self, num_feats, hidden_dim, num_classes,
                     init_type = "rnd_normal"):
        """
        initialize the weights
        """
        if init_type == "rnd_normal":
            weights = {
                "w1" : tf.Variable(tf.random_normal([num_feats, hidden_dim]), name="weight1"),
                "w2" : tf.Variable(tf.random_normal([hidden_dim, hidden_dim]), name="weight2"),
                "w3" : tf.Variable(tf.random_normal([hidden_dim, num_classes]), name="weight3")
                }

            bias = {
                "b1" : tf.Variable(tf.random_normal([1, hidden_dim]), name="bias1"),
                "b2" : tf.Variable(tf.random_normal([1, hidden_dim]), name="bias2"),
                "b3" : tf.Variable(tf.random_normal([1, num_classes]), name="bias3")
                }

        return weights, bias


def train_model():
    """
    train the model
    """
    p = Params()
    data = DataSet(100)

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
    y = tf.nn.softmax(model.network(x, weights, bias), name="softmax")

    ##
    ## define the cost function and optimization to use
    ##
    cross_entropy = -1 * tf.reduce_sum(y_ * tf.log(y))
    cost = tf.train.GradientDescentOptimizer(p.learning_rate).minimize(cross_entropy)

    ##
    ## Evaluation.
    ##
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    ##
    ## now create the session and run the train
    ##
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        ##
        ## Run the model training for num_epochsIterate and train.
        ##
        for epoch in xrange(p.num_epochs):
            for x_data, y_labels in data:
                sess.run(cost, feed_dict = {x: x_data, y_: y_labels})
            
        print "Accuracy:", sess.run(accuracy, 
                                    feed_dict = {x: test_data, 
                                                 y_: test_labels})


    ##
    ## plot the data
    ##
    data.plot()

if __name__ == '__main__':
    train_model()
    

