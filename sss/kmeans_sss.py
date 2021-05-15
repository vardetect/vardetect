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
from tensorflow.contrib.factorization import KMeansClustering
from sklearn.metrics import pairwise_distances_argmin_min
import tensorflow as tf
import numpy as np
from cfg import cfg
import tempfile, shutil, os

class KMeansSelectionStrategy(SubsetSelectionStrategy):
    def __init__(self, size, Y_vec, num_iter=None):
        self.num_clusters              = size
        self.num_iterations            = num_iter
        
        super(KMeansSelectionStrategy, self).__init__(size, Y_vec)
    
    def get_subset(self):
        points = self.Y_vec
        
        def input_fn():
            return tf.train.limit_epochs( tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)
        
        temp_name = next(tempfile._get_candidate_names())
        model_dir = os.path.join(cfg.tmpdir, temp_name) 
        kmeans = KMeansClustering(
                    num_clusters=self.num_clusters,
                    use_mini_batch=False,
                    distance_metric="squared_euclidean",
                    model_dir = model_dir
                ) # mini_batch_steps_per_iteration=self.mini_batch_steps_per_iter )

        # train
        previous_centers = None

        for i in xrange(self.num_iterations):
            kmeans.train(input_fn)
            cluster_centers = kmeans.cluster_centers()
            if previous_centers is not None:
                delta = cluster_centers - previous_centers
            if previous_centers is not None and np.sum(delta) == 0:
                break
            previous_centers = cluster_centers
        
        cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        
        uniq_clusters   = np.unique(cluster_indices)
        
        closest_data = []
        
#         print num_clusters
        
#         print cluster_centers.shape
        
#         print cluster_centers[-1,:]
        
#         print 'assigned len(np.unique(cluster_indices))', len(np.unique(cluster_indices))

        
        
        for i in uniq_clusters:
            center_vec = cluster_centers[i]
            data_idx_in_i_cluster = [ idx for idx, clu_num in enumerate(cluster_indices) if clu_num == i ]

            one_cluster_tf_matrix = np.zeros( (  len(data_idx_in_i_cluster) , cluster_centers.shape[1] ) )
            
#             print one_cluster_tf_matrix.shape
            
#             print center_vec.reshape(1,-1)
            
            for row_num, data_idx in enumerate(data_idx_in_i_cluster):
                one_row                        = points[data_idx]
                one_cluster_tf_matrix[row_num] = one_row

            closest, _ = pairwise_distances_argmin_min(center_vec.reshape(1,-1), one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_in_i_cluster[closest_idx_in_one_cluster_tf_matrix]


            closest_data.append(closest_data_row_num)
    
        shutil.rmtree(model_dir, ignore_errors=True, onerror=None)
        
        return closest_data