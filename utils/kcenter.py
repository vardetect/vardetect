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

import tensorflow as tf
import numpy as np

# Based on gist by [mbsariyildiz]
# https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
def pairwise_distances(A, B):
    
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)
    
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])
    
    D = tf.sqrt(tf.maximum(na - 2*tf.matmul(A, B, False, True) + nb, 0.0))
    return D

class KCenter(object):
    def __init__(self):
    
        self.A  = tf.placeholder(tf.float32, shape=[None,None])   # all points
        self.B  = tf.placeholder(tf.float32, shape=[None, None])  # init. center indices

        # return pairwise euclidead difference matrix
        D = pairwise_distances(self.A, self.B)

        D_min             = tf.reduce_min(D, axis=1)
        self.D_min_max    = tf.reduce_max(D_min)
        self.D_min_argmax = tf.argmax(D_min)
        
       