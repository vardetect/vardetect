import tensorflow as tf
from cnn import CNN

class DeepCNN(CNN):
    
    def __init__(self, height, width, channels, num_classes, multilabel=False, batch_size = None, is_training=True, fc_layers=[], l2_regularizer=0.001, optimizer=None, loss_fn=None, var_prefix=None, learning_rate=0.001):
        
        self.name = 'deepcnn'
        
        super(DeepCNN, self).__init__(
                height=height,
                width=width,
                channels=channels,
                num_classes=num_classes,
                multilabel=multilabel,
                batch_size=batch_size,
                is_training=is_training,
                conv_kernel_size=(3, 3),
                pool_kernel_size=(2, 2),
                num_filters=[32, 64, 128],
                fc_layers=fc_layers,
                l2_regularizer=0.001,
                convs_in_block=2,
                optimizer=optimizer,
                loss_fn=loss_fn,
                var_prefix = var_prefix,
                learning_rate = learning_rate
            )