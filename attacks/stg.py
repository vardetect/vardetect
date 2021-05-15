"""
MIT License

Copyright (c) 2016 gongzhitaao
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
import copy

def stg(model, target, k, epochs=1, clip_min=0., clip_max=1.,return_counts=False):
    """
    Grad0 method.
    :param model: model that returns the output as well as logits.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """

    model  = copy.copy(model) 
    
    def _f(xi_yi):
        xi = tf.expand_dims(xi_yi[0], axis=0)
        yi = tf.expand_dims(xi_yi[1], axis=0)
        ki = xi_yi[2]
        z, changed_count = _stg(model, xi, yi, k=ki, epochs=epochs, clip_min=clip_min, clip_max=clip_max)
        return z[0], changed_count
    
    xadv, changed_count = tf.map_fn(_f, (model.X, target, k), dtype=(tf.float32,tf.int32), back_prop=False, name='stg')
    
    if return_counts:
        return xadv, changed_count        
    else:
        return xadv
    
def _stg(model, x, y, k, epochs, clip_min, clip_max):    
    
    model.X = x
    model.build_arch()
    model.normalize_scores()
       
    xadv   = tf.identity( model.X )
    k0     = tf.argmax(tf.reshape(model.prob,[-1]))    
    
    def _cond(xadv, changed_count, i):        
        return tf.less(i, epochs)        

    def _body(xadv, changed_count, i):
        model.X = xadv
        model.build_arch()
        logits  = model.scores
               
        loss    = model.loss_fn(labels=y, logits=logits)
        dy_dx,  = tf.gradients(loss, xadv)
        
        assert dy_dx is not None
        
        x_prime = tf.nn.relu(tf.sign(dy_dx))
        x_prime = 1 - x_prime 
        
        delta_y = tf.multiply(dy_dx, x_prime-xadv)
       
        temp   = tf.reshape(delta_y,shape=(delta_y.shape[0],-1))
        result = tf.reduce_sum(tf.one_hot(indices=tf.nn.top_k(-temp,k=k)[1], depth=temp.shape[1]), axis=1)
        result = tf.reshape(result, delta_y.shape)
        
        result_prime = 1-result
        
        # what changed
        #xadv = tf.Print(xadv, [tf.boolean_mask(xadv, result), tf.boolean_mask(x_prime, result)])
        
        changed_count += tf.cast(tf.reduce_sum(result), tf.int32)
        
        # what went to 0
        #xadv = tf.Print(xadv, [tf.count_nonzero(1 - tf.boolean_mask(x_prime, result))])
        
        
        xadv = tf.stop_gradient(xadv*result_prime + x_prime*result)
        
#         assert (0 <= xadv) and ( xadv <= 1)
        
        return xadv, changed_count, i+1

    xadv, changed_count, _ = tf.while_loop(_cond, _body, (xadv, 0, 0), back_prop=False, name='_stg')
    
    return xadv,changed_count
