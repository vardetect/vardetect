from __future__ import division
import tensorflow as tf
from tfdeterminism import patch
from nn import NN

patch()

class CNN(NN):
    
    def __init__(self, height, width, channels, num_classes, multilabel=False, batch_size = None, is_training=True, conv_kernel_size=(3, 3), pool_kernel_size=(2, 2), num_filters=[32, 64, 128], fc_layers=[], l2_regularizer=0.001, convs_in_block=2, optimizer=None, loss_fn=None, var_prefix=None, learning_rate=0.001):
        
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.num_filters      = num_filters
        self.fc_layers        = fc_layers 
        self.l2_regularizer   = l2_regularizer
        self.convs_in_block   = convs_in_block
        
        assert len(conv_kernel_size) == 2
        assert len(pool_kernel_size) == 2
        
        super(CNN, self).__init__(
                height=height,
                width=width,
                channels=channels,
                num_classes=num_classes,
                multilabel=multilabel,
                batch_size=batch_size,
                is_training=is_training,
                optimizer=optimizer,
                loss_fn=loss_fn,
                var_prefix = var_prefix,
                learning_rate = learning_rate
            )
    
    def build_arch(self):
        """deepnn builds the graph for a deep net for classifying digits.
        Expects:
        self.X: an input tensor with the dimensions (N_examples, height, width, channels)
        self.model_type
        
        Creates:
        self.scores
        """
        
        self.convs = [self.X]
        
        if self.is_training:
            dropout = lambda x: tf.nn.dropout(x, self.dropout_keep_prob)
        else:
            dropout = lambda x: x
        
        for i, filters_i in enumerate(self.num_filters, start=1):

            with tf.compat.v1.variable_scope('conv%d' % i):
                for _ in range(self.convs_in_block):
                    self.convs.append(
                        tf.layers.batch_normalization(
                            tf.layers.conv2d(
                                inputs=self.convs[-1],
                                filters=filters_i,
                                kernel_size=self.conv_kernel_size,
                                padding="same",
                                activation=tf.nn.relu,
                                strides=1,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer),
                                kernel_initializer="he_normal"
                            )
                        )
                    )

            with tf.compat.v1.variable_scope('pool%d' % i):                    
                self.convs.append(
                    dropout(
                        tf.layers.max_pooling2d(
                            inputs=self.convs[-1],
                            pool_size=self.pool_kernel_size,
                            strides=2
                        )
                    )
                )
        
        self.convs = self.convs[1:]
        
        with tf.name_scope('flat'):
            dims = self.convs[-1].get_shape().as_list()
            dim  = reduce(lambda x, y: x*y, dims[1:])
            
            self.flat = tf.reshape(self.convs[-1], [-1, dim])
          
        self.fcs = [self.flat]
       
        for i, num_neurons in enumerate(self.fc_layers, start=1):
            with tf.compat.v1.variable_scope('fc%d' %i ): 
                    self.fcs.append(dropout(tf.layers.dense(self.fcs[-1], num_neurons, activation=tf.nn.relu)))
            
        with tf.compat.v1.variable_scope('scores'):
            self.scores = tf.layers.dense(self.fcs[-1], self.num_classes)


    def print_arch( self ):
        
        print self.X
        print self.labels
        
        for conv in self.convs:
            print conv
          
        for fc in self.fcs:
            print fc
        
        print self.scores
        print self.prob
        print self.predictions
