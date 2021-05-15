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
import numpy as np

def pgd(model, epochs=10, y=None, eps=0.1, step_size=None, sign=True, lower_bound=0., upper_bound=1.,perturbation_multiplier=1, random_step=False):
    """
    PGD: Iterative Fast gradient method.
    :param model: model that returns the output as well as logits.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    model = copy.copy(model)
    
    if y is None:
        y = model.predictions_one_hot
                       
    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity
    
#     if step_size is None:    
    def _f(xi_yi):
        xi = tf.expand_dims(xi_yi[0], axis=0)
        yi = tf.expand_dims(xi_yi[1], axis=0)
        step_size = eps/float(epochs)           
        
        print "Step size of perturbation: {}" .format(step_size) 
        
        z = _pgd(model, xi, yi, eps, step_size, epochs, noise_fn, lower_bound, upper_bound,perturbation_multiplier,random_step)
        return z[0]

    xadv = tf.map_fn(_f, (model.X, y), dtype=(tf.float32), back_prop=False, name='pgd')        

        #     else:        
#         def _f(xi_yi):
#             xi = tf.expand_dims(xi_yi[0], axis=0)
#             yi = tf.expand_dims(xi_yi[1], axis=0)
#             ei = xi_yi[2]
#             z = _pgd(model, xi, yi, epsilon, step_size, ei, noise_fn, lower_bound, upper_bound,perturbation_multiplier)
#             return z[0]
          
#         xadv = tf.map_fn(_f, (model.X, y, epochs), dtype=(tf.float32), back_prop=False, name='pgd')
                
    return xadv

def _pgd(model, x, y, eps, step_size, epochs, noise_fn, lower_bound, upper_bound,perturbation_multiplier, random_step):        

    clip_min = tf.maximum(x - eps,lower_bound)
    clip_max = tf.minimum(x + eps, upper_bound)
    
    if random_step:
        xadv = x + tf.random_uniform(tf.shape(x), -eps, eps)
        xadv = tf.clip_by_value(xadv, clip_min, clip_max)
    else:
        xadv = tf.identity(x)
    
    k0   = tf.argmax(tf.reshape(y,[-1]))
    
    def _cond(xadv, i):
    
        model.X = xadv
        model.build_arch()
        model.normalize_scores()
        
        k  = tf.argmax(tf.reshape(model.prob,[-1]))       
        
        if perturbation_multiplier == 1:
            return tf.logical_and( tf.less(i, epochs),
                                   tf.equal(k0, k)
                                 )        
        else:
            return tf.logical_and( tf.less(i, epochs),
                                   tf.not_equal(k0, k)
                                 )        
            
    def _body(xadv, i):
        model.X = xadv
        model.build_arch()
        logits  = model.scores
        loss    = model.loss_fn(labels=y, logits=logits)
        dy_dx,  = tf.gradients(loss, xadv)
        
        assert dy_dx is not None
        
        xadv    = tf.stop_gradient(xadv + (perturbation_multiplier*step_size*noise_fn(dy_dx) ) )
        xadv    = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='_pgd')
    
    return xadv
