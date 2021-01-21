from __future__ import division
import tensorflow as tf
from nn import NN
import numpy as np
 
class NewVAE(NN):
    
    def __init__(self, height, width, channels, num_classes=None, z_size=32, decoder_units=49, batch_size = None, is_training=True, conv_kernel_size=(3, 3), pool_kernel_size=(2, 2), fc_layers=[], l2_regularizer=0.001, convs_in_block=1, optimizer=None, learning_rate = 0.001, loss_fn=None, activation=None, train_epochs = 400, random_draws = 1, noise_mean=100.0 ):
        
        self.X                 = tf.compat.v1.placeholder(tf.float32, shape=(None, height, width, channels), name='X')
        self.labels            = tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels')
        self.dropout_keep_prob = tf.compat.v1.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), tuple(),name='dropout_keep_prob')        
        
        
        self.height           = height
        self.width            = width
        self.channels         = channels
        self.batch_size       = batch_size
        self.random_draws     = random_draws
        
        self.var_prefix       = 'vae'        
        self.name             = 'vae'
        
        self.learning_rate    = learning_rate
        
        self.decoder_units    = decoder_units
        self.z_size           = z_size
        self.optimizer        = optimizer
        self.channels         = channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.fc_layers        = fc_layers 
        self.l2_regularizer   = l2_regularizer
        self.convs_in_block   = convs_in_block        
        self.is_training      = is_training                
        self.train_epochs     = train_epochs
        
        self.noise_mean       = noise_mean
        
        if activation is None:
            self.activation = tf.nn.leaky_relu
        else: 
            self.activation = activation
                
        assert len(conv_kernel_size) == 2
        assert len(pool_kernel_size) == 2
        
        self.setup_model()
        
        if is_training:
            assert( self.random_draws == 1 )
            self.setup_training()
    
    
    def setup_model( self ):
        self.build_arch()
        self.setup_loss()
        self.setup_recon()
    
    
    def build_arch(self):
        def _tile(x):
            return tf.tile( tf.expand_dims(x, axis=0)  , (self.random_draws,1,1,1) )

        self.z, self.mean, self.std_dev = self.encoder(self.X)
        self.x_hat                      = self.decoder(self.z)
        

    def setup_recon( self ):
        rloss = tf.reduce_sum( tf.reshape(tf.math.squared_difference(self.x_hat, self.X), (-1, self.random_draws, self.height*self.width*self.channels)), -1)
        
        self.best_idx            = tf.argmin(rloss, axis=1)
        self.best_recon_loss_mse = tf.reduce_min(rloss, axis=1)
            
        x_hat_recon_reshaped = tf.reshape( self.x_hat, (-1, self.random_draws, self.height, self.width, self.channels ) )
        
        def _slice(xi_yi):
            xi = xi_yi[0]
            yi = xi_yi[1]    
            return tf.gather(xi,yi)

        self.decoded = tf.map_fn(_slice, (x_hat_recon_reshaped, self.best_idx), dtype=(tf.float32), back_prop=False, name='slicer')
        
    def encoder(self, X):
        raise NotImplementedError("Subclasses must override")
        
                    
    def decoder( self, z ):
        raise NotImplementedError("Subclasses must override")
    
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/sum_loss', self.sum_loss))
        train_summary.append(tf.summary.scalar('train/mean_loss', self.mean_loss))
        self.train_summary = tf.summary.merge(train_summary)        
        
        
    def setup_loss(self):
        with tf.name_scope('loss'):
            flat_input   = tf.reshape(self.X, [-1, self.height * self.width * self.channels])
            flat_output  = tf.reshape(self.x_hat, [-1, self.height * self.width * self.channels])  
            self.recon_loss_mse = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            self.recon_loss_mae = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            self.recon_loss_ce = tf.reduce_sum(flat_input * -tf.math.log(flat_output) + (1-flat_input) * -tf.math.log(1-flat_output), 1)
            self.recon_loss    = self.recon_loss_mse
            self.latent_loss_m1   = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.std_dev) - tf.math.log(tf.square(self.std_dev)) - 1, 1)
            self.latent_loss_m2   = 0.5 * tf.reduce_sum(tf.square(self.mean - self.noise_mean ) + tf.square(self.std_dev) - tf.math.log(tf.square(self.std_dev)) - 1, 1)            
            self.latent_loss      = tf.where(tf.equal(tf.argmax(self.labels, axis=1), 0), self.latent_loss_m1, self.latent_loss_m2)

            self.sum_recon_loss  = tf.reduce_sum( self.recon_loss)
            self.sum_latent_loss = tf.reduce_sum( self.latent_loss)
            
            self.mean_recon_loss  = tf.reduce_mean( self.recon_loss )
            self.mean_latent_loss = tf.reduce_mean( self.latent_loss )
            
            self.sum_loss  = tf.reduce_sum(self.recon_loss_mse + self.latent_loss)
            self.mean_loss = tf.reduce_mean(self.recon_loss_mse + self.latent_loss)
    
        
    def filter_predictions( self, predictions, prob, threshold, num_classes ):
        mask     =  self.recon_loss_mse < threshold         
        paddings = tf.constant([[0, 0,], [0, 1]])
        mask_prob = tf.where(mask, tf.pad(prob, paddings, "CONSTANT"), tf.one_hot(predictions, num_classes) )
        mask     = tf.cast(mask, tf.int64 )
        
        return mask*predictions + (1-mask)* (num_classes-1), mask_prob
                    
        
    def is_multilabel(self):
        return False

class NewHSVNVAE28(NewVAE):
    def __init__(self, height, width, channels, num_classes=None, z_size=32, decoder_units=49, batch_size = None, is_training=True, conv_kernel_size=(3, 3), pool_kernel_size=(2, 2), fc_layers=[], l2_regularizer=0.001, convs_in_block=1, optimizer=None, learning_rate = 0.001, loss_fn=None, activation=None, train_epochs=100, random_draws=1, noise_mean=5.0 ):
        super(NewHSVNVAE28, self).__init__(
                                        height  = height,
                                        width   = width,
                                        channels=channels,
                                        z_size  = z_size,
                                        batch_size=batch_size,
                                        is_training=is_training,
                                        conv_kernel_size=conv_kernel_size,
                                        pool_kernel_size=pool_kernel_size,
                                        fc_layers=fc_layers,
                                        l2_regularizer=l2_regularizer,
                                        convs_in_block=convs_in_block,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        activation = activation,
                                        learning_rate = learning_rate,
                                        train_epochs=train_epochs, 
                                        random_draws = random_draws,
                                        noise_mean   = noise_mean,
                                        num_classes = num_classes
                                     )

    def encoder(self, X):
        with tf.compat.v1.variable_scope("Encoder", reuse = tf.compat.v1.AUTO_REUSE):
            self.conv1 = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv3 = tf.layers.conv2d(self.conv2, filters=128, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv4 = tf.layers.conv2d(self.conv3, filters=256, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer)) 
            
            self.convs = []
            self.convs.append(self.conv1)
            self.convs.append(self.conv2)
            self.convs.append(self.conv3)
            
            self.flat   = tf.layers.flatten(self.conv3)
            self.dense1 = tf.layers.dense(self.flat, units=512, activation=self.activation)
            
            self.mean    = tf.layers.dense(self.dense1, units=self.z_size, name='mean')
            self.std_dev = tf.nn.softplus(tf.layers.dense(self.dense1, units=self.z_size), name='std_dev')

            epsilon = tf.random.normal(tf.stack([tf.shape(self.X)[0], self.z_size]), name='epsilon')
            self.z  = self.mean + tf.multiply(epsilon, self.std_dev)

        return self.z, self.mean, self.std_dev        
        
                    
    def decoder( self, z ):
        with tf.compat.v1.variable_scope("Decoder", reuse= tf.compat.v1.AUTO_REUSE):
            self.decoder_h1 = tf.layers.dense(z, units=512, activation=self.activation)
            self.decoder_dense = []
            
            self.decoder_dense.append( self.decoder_h1 )
            self.decoder_h1_reshape = tf.reshape(self.decoder_h1, [-1, 1, 1, 512])
            
            self.dconv1 = tf.layers.conv2d_transpose(self.decoder_h1_reshape, 256, 4, strides=2, activation = self.activation, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))            
            self.dconv2 = tf.layers.conv2d_transpose(self.dconv1, 128, 4, strides=2, activation = self.activation, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.dconv3 = tf.layers.conv2d_transpose(self.dconv2, 64, 4, strides=1, activation = self.activation, padding='valid', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.dconv4 = tf.layers.conv2d_transpose(self.dconv3, 32, 4, strides=2, activation = self.activation, padding='same')

            self.dconvs = []
            self.dconvs.append(self.dconv1)
            self.dconvs.append(self.dconv2)
            self.dconvs.append(self.dconv3)
            self.dconvs.append(self.dconv4)
            
            self.x_hat = tf.layers.conv2d_transpose(self.dconv4, filters=self.channels, kernel_size=4, strides=2, activation=tf.nn.sigmoid, padding='same')
            
        return self.x_hat
        
    def setup_loss(self):
        with tf.name_scope('loss'):
            
            flat_input   = tf.reshape(self.X, [-1, self.height * self.width * self.channels])
            flat_output  = tf.reshape(self.x_hat, [-1, self.height * self.width * self.channels])                        
            
            self.recon_loss_mse = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            self.recon_loss_mae = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            
            self.recon_loss_ce = tf.reduce_sum(flat_input * -tf.math.log(flat_output) + (1-flat_input) * -tf.math.log(1-flat_output), 1)
            self.recon_loss    = self.recon_loss_mse
            
            self.latent_loss_m1   = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.std_dev) - tf.math.log(tf.square(self.std_dev)) - 1, 1)
            self.latent_loss_m2   = 0.5 * tf.reduce_sum(tf.square(self.mean - self.noise_mean ) + tf.square(self.std_dev) - tf.math.log(tf.square(self.std_dev)) - 1, 1)            
            
            self.latent_loss      = tf.where(tf.equal(tf.argmax(self.labels, axis=1), 0), self.latent_loss_m1, self.latent_loss_m2)

            self.sum_recon_loss  = tf.reduce_sum( self.recon_loss)
            self.sum_latent_loss = tf.reduce_sum( self.latent_loss)
            
            self.mean_recon_loss  = tf.reduce_mean( self.recon_loss )
            self.mean_latent_loss = tf.reduce_mean( self.latent_loss )
            
            self.sum_loss  = tf.reduce_sum(self.recon_loss_mse + 0.5 * self.latent_loss)
            self.mean_loss = tf.reduce_mean(self.recon_loss_mse + 0.5 * self.latent_loss)
    
    
    def print_arch( self ):
        print self.X
        print self.labels
        
        for conv in self.convs:
            print conv
        
        print self.dense1
        print self.mean
        print self.std_dev
        print self.z
        
        for d in self.decoder_dense:
            print d
                
        for dconv in self.dconvs:
            print dconv
                
        print self.x_hat

class NewHSVNVAE(NewVAE):
    def __init__(self, height, width, channels, num_classes=None, z_size=32, decoder_units=49, batch_size = None, is_training=True, conv_kernel_size=(3, 3), pool_kernel_size=(2, 2), fc_layers=[], l2_regularizer=0.001, convs_in_block=1, optimizer=None, learning_rate = 0.001, loss_fn=None, activation=None, train_epochs=600, random_draws=1, noise_mean=10.0 ):
        super(NewHSVNVAE, self).__init__(
                                        height  = height,
                                        width   = width,
                                        channels=channels,
                                        z_size  = z_size,
                                        batch_size=batch_size,
                                        is_training=is_training,
                                        conv_kernel_size=conv_kernel_size,
                                        pool_kernel_size=pool_kernel_size,
                                        fc_layers=fc_layers,
                                        l2_regularizer=l2_regularizer,
                                        convs_in_block=convs_in_block,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        activation = activation,
                                        learning_rate = learning_rate,
                                        train_epochs=train_epochs, 
                                        random_draws = random_draws,
                                        noise_mean   = noise_mean,
                                        num_classes = num_classes
                                     )

    def encoder(self, X):
        with tf.variable_scope("Encoder", reuse = tf.AUTO_REUSE):
            self.conv1 = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv3 = tf.layers.conv2d(self.conv2, filters=128, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.conv4 = tf.layers.conv2d(self.conv3, filters=256, kernel_size=4, strides=2, padding='same', activation=self.activation, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer)) 
            
            self.convs = []
            self.convs.append(self.conv1)
            self.convs.append(self.conv2)
            self.convs.append(self.conv3)
            
            self.flat   = tf.layers.flatten(self.conv3)
            self.dense1 = tf.layers.dense(self.flat, units=512, activation=self.activation)
            
            # Local latent variables
            self.mean    = tf.layers.dense(self.dense1, units=self.z_size, name='mean')
            self.std_dev = tf.nn.softplus(tf.layers.dense(self.dense1, units=self.z_size), name='std_dev')  # softplus to force >0

            # Reparametrization trick
            epsilon = tf.random_normal(tf.stack([tf.shape(self.X)[0], self.z_size]), name='epsilon')
            self.z  = self.mean + tf.multiply(epsilon, self.std_dev)

        return self.z, self.mean, self.std_dev        
        
                    
    def decoder( self, z ):
        with tf.variable_scope("Decoder", reuse= tf.AUTO_REUSE):
            self.decoder_h1 = tf.layers.dense(z, units=512, activation=self.activation)
            self.decoder_dense = []
            
            self.decoder_dense.append( self.decoder_h1 )
            self.decoder_h1_reshape = tf.reshape(self.decoder_h1, [-1, 1, 1, 512])
            
            self.dconv1 = tf.layers.conv2d_transpose(self.decoder_h1_reshape, 256, 4, strides=2, activation = self.activation, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))            
            self.dconv2 = tf.layers.conv2d_transpose(self.dconv1, 128, 4, strides=2, activation = self.activation, padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.dconv3 = tf.layers.conv2d_transpose(self.dconv2, 64, 4, strides=1, activation = self.activation, padding='valid', kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            self.dconv4 = tf.layers.conv2d_transpose(self.dconv3, 32, 4, strides=2, activation = self.activation, padding='same')

            self.dconvs = []
            self.dconvs.append(self.dconv1)
            self.dconvs.append(self.dconv2)
            self.dconvs.append(self.dconv3)
            self.dconvs.append(self.dconv4)
            
            for dconv in self.dconvs:
                print(dconv)
            
            self.x_hat = tf.layers.conv2d_transpose(self.dconv4, filters=self.channels, kernel_size=4, strides=2, activation=tf.nn.sigmoid, padding='same')
            
            print(self.x_hat)
            
        return self.x_hat
        
    def setup_loss(self):
        with tf.name_scope('loss'):
            
            flat_input   = tf.reshape(self.X, [-1, self.height * self.width * self.channels])
            flat_output  = tf.reshape(self.x_hat, [-1, self.height * self.width * self.channels])                        
            
            self.recon_loss_mse = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            self.recon_loss_mae = tf.reduce_sum( tf.square(flat_input-flat_output), axis=1 )
            
            self.recon_loss_ce = tf.reduce_sum(flat_input * -tf.log(flat_output) + (1-flat_input) * -tf.log(1-flat_output), 1)
            self.recon_loss    = self.recon_loss_mse
            
            self.latent_loss_m1   = 0.5 * tf.reduce_sum(tf.square(self.mean) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)
            self.latent_loss_m2   = 0.5 * tf.reduce_sum(tf.square(self.mean - self.noise_mean ) + tf.square(self.std_dev) - tf.log(tf.square(self.std_dev)) - 1, 1)            
            
            self.latent_loss      = tf.where(tf.equal(tf.argmax(self.labels, axis=1), 0), self.latent_loss_m1, self.latent_loss_m2)

            self.sum_recon_loss  = tf.reduce_sum( self.recon_loss)
            self.sum_latent_loss = tf.reduce_sum( self.latent_loss)
            
            self.mean_recon_loss  = tf.reduce_mean( self.recon_loss )
            self.mean_latent_loss = tf.reduce_mean( self.latent_loss )
            
            self.sum_loss  = tf.reduce_sum(self.recon_loss_mse + 0.5 * self.latent_loss)
            self.mean_loss = tf.reduce_mean(self.recon_loss_mse + 0.5 * self.latent_loss)
    
    
    def print_arch( self ):
        print self.X
        print self.labels
        
        for conv in self.convs:
            print conv
        
        print self.dense1
        print self.mean
        print self.std_dev
        print self.z
        
        for d in self.decoder_dense:
            print d
                
        for dconv in self.dconvs:
            print dconv
                
        print self.x_hat