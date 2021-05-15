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
from adversarial_sss import AdversarialSelectionStrategy
from tensorflow.contrib.factorization import KMeansClustering
import tensorflow as tf
import numpy as np
import math
from cfg import config, cfg
from utils.kcenter import KCenter
import tempfile, shutil, os


class KCenterAdvesarial(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, init_cluster, X_vec, sess, copy_model):
        self.init_cluster    = init_cluster
        self.X_vec        = X_vec
        self.sess         = sess
        self.copy_model   = copy_model
        assert( len(X_vec) == len(Y_vec) )
        
        super(KCenterAdvesarial, self).__init__(size, Y_vec)
        
        
    def get_subset(self):
        
        X = self.Y_vec
        Y = self.init_cluster
        
        batch_size = 100*cfg.batch_size
        
        def input_fn():
            return tf.train.limit_epochs( tf.convert_to_tensor(X, dtype=tf.float32), num_epochs=1)
        
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

        temp_name = next(tempfile._get_candidate_names())
        model_dir = os.path.join(cfg.tmpdir, temp_name) 
        
        
        print X.shape, len(points)
        
        kmeans    = KMeansClustering( num_clusters=len(points), use_mini_batch=False, distance_metric="squared_euclidean",model_dir = model_dir, initial_clusters=X[points])
        
        kmeans.train(input_fn)
        
#         print "kmeans.cluster_centers == points", np.allclose(kmeans.cluster_centers, X[points])        
#         print kmeans.cluster_centers
#         print X[points]
        
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        
        uniq_clusters   = np.unique(cluster_indices)
        z               = []
        
        for i in uniq_clusters:
            data_idx_in_i_cluster = [ idx for idx, clu_num in enumerate(cluster_indices) if clu_num == i ]
            
            print data_idx_in_i_cluster
            
            one_cluster_tf_matrix = np.zeros( [len(data_idx_in_i_cluster)] + list(self.X_vec.shape[1:]) ) 

            for row_num, data_idx in enumerate(data_idx_in_i_cluster):
                one_row                        = self.X_vec[data_idx]
                one_cluster_tf_matrix[row_num] = one_row
                 
            sss = AdversarialSelectionStrategy(1, None, one_cluster_tf_matrix, self.sess, self.copy_model, K=len(one_cluster_tf_matrix))

            closest_idx_in_one_cluster_tf_matrix = sss.get_subset()[0]
            
#             print closest_idx_in_one_cluster_tf_matrix, len(data_idx_in_i_cluster)

            closest_data_row_num = data_idx_in_i_cluster[closest_idx_in_one_cluster_tf_matrix]

            z.append(closest_data_row_num)
                
        return z