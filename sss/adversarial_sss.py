"""
MIT License

Copyright (c) 2019 Soham Pal, Yash Gupta, Aditya Kanade, Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.
Modified in 2019 by Yash Gupta, Soham Pal, Aditya Kanade, Shirish Shevade. Indian Institute of Science.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from base_sss import SubsetSelectionStrategy
import base_sss
from attacks.deepfool import deepfool
from cfg import cfg, config
import tensorflow as tf
import numpy as np
import random

class AdversarialSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, X_vec, sess, copy_model, K=10000, perm=None, epochs=2):
        self.copy_model = copy_model
        self.X_vec = X_vec
        self.sess  = sess
        self.K     = K
        self.perm = perm
        if self.K > len(self.X_vec):
            self.K = len(self.X_vec)
        
        print "number of epochs for adversarial strategy", epochs
        
        with tf.variable_scope("copy_model", reuse=True):
            self.xadv = deepfool(copy_model, epochs=epochs)  # Ideally batch=False and epochs>=10

        # difference as an $L_2$ norm
        self.diff = tf.reduce_sum(
                        tf.reduce_sum(
                            tf.reduce_sum(
                                tf.squared_difference(self.xadv, copy_model.X),
                                axis=1
                            ),
                            axis=1
                        ),
                    axis=1)

        super(AdversarialSelectionStrategy, self).__init__(size, Y_vec)
        
    
    def get_subset(self):
        diffs = np.ones(len(self.X_vec)) * np.infty
        
        if self.perm is None:
            p = base_sss.sss_rs.permutation(len(self.X_vec))
        else:
            p = self.perm
            
        rev_p = np.argsort(p)

        self.X_vec = self.X_vec[p]

        for start in range(0, self.K, cfg.batch_size):
            end = start + cfg.batch_size

            if end > len(self.X_vec):
                end = len(self.X_vec)

            diffs[start:end] = list(self.sess.run(self.diff, {self.copy_model.X: self.X_vec[start:end], self.copy_model.dropout_keep_prob: 1.0}))

        self.X_vec = self.X_vec[rev_p]
        diffs = diffs[rev_p]

        assert len(diffs) == len(self.X_vec)
#         print 'Sorted diffs:', np.sort(diffs)[:self.size]
#         print 'ArgSorted diffs:', np.argsort(diffs)[:self.size]

        return np.argsort(diffs)[:self.size]