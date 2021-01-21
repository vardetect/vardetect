from __future__ import division
import tensorflow as tf
from nn import NN

class DefenseModel(NN):
    
    def __init__(self, true_model, filter_model, threshold, defender_type, use_recon_input):
        assert(true_model.is_training == False)
        assert(filter_model.is_training == False)
        
        self.true_model         = true_model
        self.filter_model       = filter_model        
        self.defender_type      = defender_type
        self.use_recon_input    = use_recon_input

        if self.use_recon_input:        
            self.X                        = filter_model.X         
            true_model.X                  = filter_model.decoded        
            true_model.setup_model()
        else:                        
            self.X                        = true_model.X         
            self.filter_model.X           = self.X        
            self.filter_model.setup_model()
        
        self.batch_size               = true_model.get_batch_size()
        self.labels                   = true_model.labels
        self.dropout_keep_prob        = true_model.dropout_keep_prob        
        self.num_classes              = true_model.get_num_classes() + 1                 
        self.multilabel               = False
        
        self.true_predictions         = true_model.predictions
        self.true_predictions_one_hot = true_model.predictions_one_hot        
        self.true_prob                = true_model.prob
        
        self.predictions, self.prob   = filter_model.filter_predictions( self.true_predictions, self.true_prob, threshold, self.num_classes)

        self.predictions_one_hot      = tf.one_hot(self.predictions, self.num_classes)
        
        
        self.measure_accuracy()
        
                       