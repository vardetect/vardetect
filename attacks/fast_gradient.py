"""
MIT License

Copyright (c) 2016 gongzhitaao
Modified in 2019 by Soham Pal, Yash Gupta, Aditya Kanade, Shirish Shevade, Vinod Ganapathy. Indian Institute of Science.
Modified in 2019 by Yash Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.

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
from models.vae import NewVAE


def fgm(model, y=None, eps=0.01, epochs=1, sign=True, clip_min=0., clip_max=1.,perturbation_multiplier=1):
    """
    Fast gradient method.
    :param model: model that returns the output as well as logits.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.
    :return: A tensor, contains adversarial samples for each input.
    """
    model = copy.copy(model)
    
    original_model_X = model.X
    xadv    = tf.identity( model.X )
    
    if y is None:
        y = model.predictions_one_hot

    if sign:
        noise_fn = tf.sign
    else:
        noise_fn = tf.identity

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        model.X = xadv
        model.build_arch()
        
        if type(model) == NewVAE:
            print('Working with VAE...')
            loss    = tf.losses.mean_squared_error(y, model.mean)
        else:
            logits  = model.scores
            loss    = model.loss_fn(labels=y, logits=logits)
        
        dy_dx,  = tf.gradients(loss, xadv)
        
        assert dy_dx is not None
        
        xadv    = tf.stop_gradient(xadv + (perturbation_multiplier*eps*noise_fn(dy_dx)) )
        xadv    = tf.clip_by_value(xadv, clip_min, clip_max)
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='fast_gradient')
    
    return xadv
