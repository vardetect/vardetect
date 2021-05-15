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
import tensorflow as tf
import numpy as np
import math
from cfg import config, cfg
from utils.kcenter import KCenter

class KCenterGreedyApproach(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, init_cluster):
        self.init_cluster    = init_cluster
        
        super(KCenterGreedyApproach, self).__init__(size, Y_vec)
        
        
    def get_subset(self):
#         A = np.vstack([self.Y_vec, self.init_cluster])
#         B = np.arange(self.Y_vec.shape[0], A.shape[0] ) 
        
        X = self.Y_vec
        Y = self.init_cluster
        
        batch_size = 100*cfg.batch_size
        
        n_batches  = int(math.ceil(len(X)/float(batch_size)))
        
        m = KCenter()
        
        points = []
        
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for _ in range(self.size):
                p = []
                q = []
                for i in range(n_batches):
                    start = i*batch_size
                    end   = i*batch_size + batch_size 
                    X_b   = X[start:end]
                    D_argmax_val, D_min_max_val = sess.run( [m.D_min_argmax, m.D_min_max], feed_dict={ m.A: X_b, m.B:Y } )
#                     print "D val :"
#                     print D_val
#                     print D_min_max_val
#                     print D_argmax_val

                    p.append(D_argmax_val)
                    q.append(D_min_max_val)

                b_indx = np.argmax(q)
                indx   = b_indx*batch_size + p[b_indx]
                Y      = np.vstack([Y, X[indx]])
                
                points.append(indx)
            
        return points