from __future__ import division
import tensorflow as tf
from tfdeterminism import patch

patch()

class NN(object):
    
    def __init__(self, height, width, channels, num_classes, multilabel=False, batch_size = None, is_training=True, optimizer=None, loss_fn=None, learning_rate=0.001, var_prefix=None):
        
        if width == None or channels == None:
            self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, height), name='X')
        else:
            self.X = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, channels), name='X')
        
        self.labels            = tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        self.dropout_keep_prob = tf.compat.v1.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), tuple(),name='dropout_keep_prob')
        
        self.var_prefix  = var_prefix
        
        self.num_classes = num_classes
        self.is_training = is_training
        self.height      = height
        self.width       = width
        self.channels    = channels
        self.multilabel  = multilabel
        self.batch_size  = batch_size
        self.optimizer   = optimizer
        self.learning_rate = learning_rate
        self.loss_fn     = loss_fn
        self.setup_model()
        if is_training:
            self.setup_training()
        
    def setup_model( self ):
        self.build_arch()
        self.normalize_scores()
        self.measure_accuracy()
        self.setup_loss()
        
    def get_batch_size(self):
        return self.batch_size
        
    def measure_accuracy(self):
        with tf.name_scope("accuracy"):
            if self.multilabel:
                ground_truth              = tf.round(self.labels)
            else: 
                ground_truth              = tf.argmax(self.labels, axis=-1)
            
            correct_prediction            = tf.equal(ground_truth, self.predictions)
            self.sum_correct_prediction   = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy                 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def is_multilabel(self):
        return self.multilabel
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_num_channels(self):
        return self.channels    
    
    def build_arch(self):
        raise NotImplementedError("Subclasses must override")
        
    def setup_loss(self):
 
        with tf.name_scope('loss'):
            if self.loss_fn is None:
                if self.multilabel:
                    self.loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
                else:
                    self.loss_fn = tf.nn.softmax_cross_entropy_with_logits

                self.cross_entropy = self.loss_fn(labels=tf.stop_gradient(self.labels), logits=self.scores)
            else:
                self.cross_entropy, _ = self.loss_fn(self.labels, self.prob)
            
            
            self.entropy       = tf.nn.softmax_cross_entropy_with_logits(labels=self.prob, logits=self.scores)
            self.sum_loss      = tf.reduce_sum(self.cross_entropy)
            self.mean_loss     = tf.reduce_mean(self.cross_entropy)
    
    def setup_training(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self.optimizer is None:
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            print "Adam Learning rate:", self.learning_rate
            
        self.train_op    = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)
        self._summary()
    
    def normalize_scores(self):
        assert self.scores is not None
        
        with tf.name_scope('output'):
            if self.multilabel:
                self.prob        = tf.nn.sigmoid(self.scores, name="prob")
                self.predictions = tf.round( self.prob )
                self.predictions_one_hot = self.predictions
            else:
                self.prob                = tf.nn.softmax(self.scores, axis=-1, name="prob")
                self.predictions         = tf.argmax(self.prob, axis = -1)
                self.predictions_one_hot = tf.one_hot(self.predictions, self.num_classes) 

    def print_trainable_parameters( self ):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print "Total trainable parameters ", total_parameters        
    
    def _summary(self):
        train_summary = []
        train_summary.append(tf.compat.v1.summary.scalar('train/mean_loss', self.mean_loss))
        train_summary.append(tf.compat.v1.summary.scalar('train/accuracy', self.accuracy))
        self.train_summary = tf.compat.v1.summary.merge(train_summary)


    def print_arch( self ):
        raise NotImplementedError("Subclasses must override")

    def get_graph(self):
        return tf.get_default_graph()