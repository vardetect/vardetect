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
import pandas as pd
import time, os, shutil, glob, gc, tempfile, sys, math
from cfg import cfg, config
from dsl.dsl_marker_v2 import DSLMarker, collect_aux_data
from sss.base_sss import SubsetSelectionStrategy
from sss.random_sss import RandomSelectionStrategy
from sss.adversarial_sss import AdversarialSelectionStrategy
from sss.balancing_sss import BalancingSelectionStrategy
from sss.uncertainty_sss import UncertaintySelectionStrategy
from sss.kmeans_sss import KMeansSelectionStrategy
from sss.kcenter_sss import KCenterGreedyApproach
from sss.kcenter_adv_sss import KCenterAdvesarial
from sklearn.metrics import f1_score
from attacks.fast_gradient import fgm
from attacks.pgd import pgd
from attacks.cw import cw
from utils.helper import *
from attacks.grad0 import grad0
from attacks.deepfool import deepfool
from dsl.base_dsl import one_hot_labels  
import imageio, copy
from sklearn.svm import SVC
import pickle, random
import joblib
from dsl.uniform_dsl import UniformDSL

def train_model(model, train_dsl, val_dsl, logdir=None, use_early_stop=True):
    "Trains the model and saves the best model in logdir"
    num_batches_tr  = train_dsl.get_num_batches()
    
    if logdir is not None:
        train_writer = tf.summary.FileWriter(logdir)
        train_writer.add_graph(model.get_graph()) 
    
    orig_var_list = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    saver         = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    with tf.compat.v1.Session(config =config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        curr_acc = None
        best_acc = None
        no_improvement = 0
        
        for epoch in range(1, cfg.num_epochs+1):
            epoch_time = time.time()
            t_loss = []
            for b_tr in range(num_batches_tr): 
                
                if 'text' in model.name: 
                    trX, trY = train_dsl.load_next_batch(b_tr, return_lengths=False)
                else:
                    trX, trY = train_dsl.load_next_batch(b_tr)
                    
                feed_dict = { model.X: trX, model.labels: trY, model.dropout_keep_prob: cfg.dropout_keep_prob }                
                
                global_step, _, summary_str, loss = sess.run([ model.global_step,
                                                               model.train_op,
                                                               model.train_summary,
                                                               model.mean_loss],
                                                               feed_dict
                                                            )
                t_loss.append(loss)    
                
                if logdir is not None:
                    train_writer.add_summary(summary_str, global_step)
                    train_writer.flush()
        
        
            if use_early_stop:
                if epoch % cfg.evaluate_every == 0:

                    curr_acc = compute_evaluation_measure(model, sess, val_dsl, model.sum_correct_prediction)

                    if best_acc is None or curr_acc > best_acc :
                        best_acc = curr_acc

                        if logdir is not None:
                            save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                            print "Model saved in path: %s" % save_path

                        print "[BEST]",

                        no_improvement = 0
                    else:
                        no_improvement += 1

                        if (no_improvement % cfg.early_stop_tolerance) == 0:
                            break
                        
                    print "Step: {} \tValAccuracy: {} \tTrainLoss: {}" .format( global_step, curr_acc, np.mean(t_loss))  
            else:
                print "Step: {}\tTrainLoss: {}" .format( global_step, np.mean(t_loss))  
                                

            print "End of epoch {} (took {} minutes)." .format(epoch, round((time.time() - epoch_time)/60, 2)) 

            
def train_vae(model, train_dsl, val_dsl, logdir, logdir_true):
    "Trains the model and saves the best model in logdir"
    num_batches_tr  = train_dsl.get_num_batches()
    num_batches_val = val_dsl.get_num_batches()
    
    num_samples_val = val_dsl.get_num_samples()
    
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(model.get_graph()) 
    
    if logdir_true is not None:
        orig_var_list = [v for v in tf.compat.v1.global_variables() if v.name.startswith('true_model')]
        orig_saver    = tf.compat.v1.train.Saver(var_list=orig_var_list)
    
    saver    = tf.compat.v1.train.Saver()
    
    path     = logdir + '/images'
    create_dirs([path])    
    
    with tf.compat.v1.Session(config =config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        if logdir_true is not None:
            orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
        
        curr_loss = None
        best_loss = None
        no_improvement = 0
        
        for epoch in range(1, model.train_epochs+1):
            epoch_time = time.time()
            t_loss     = []
            r_loss     = []
            l_loss     = []
            for b_tr in range(num_batches_tr): 

                trX, trY = train_dsl.load_next_batch(b_tr)
                
                global_step, _, summary_str, loss, recon_loss, latent_loss = sess.run([
                                                 model.global_step,
                                                 model.train_op,
                                                 model.train_summary,
                                                 model.mean_loss,
                                                 model.mean_recon_loss,
                                                 model.mean_latent_loss
                                              ],
                                              feed_dict={
                                                  model.X: trX,
                                                  model.dropout_keep_prob: cfg.dropout_keep_prob,
                                                  model.labels: trY
                                              })
                t_loss.append(loss)    
                r_loss.append(recon_loss)
                l_loss.append(latent_loss)
                
                train_writer.add_summary(summary_str, global_step)
                train_writer.flush()
        
            curr_loss = compute_evaluation_measure(model, sess, val_dsl, model.sum_loss)

            
            if epoch == 1:
                valX, _ = val_dsl.load_next_batch(1)                
                img_grid = merge(valX[0:144], [12,12])     
                img_grid = np.array(img_grid * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(path, 'og_img.png'), img_grid)
            
            if epoch % 10 == 0:
                valX, _ = val_dsl.load_next_batch(1)
                decoded = sess.run(model.decoded, { model.X: valX  })
                         
                img_grid = merge(decoded[0:144], [12,12])
                img_grid = np.array(img_grid * 255).astype(np.uint8)                    
                imageio.imwrite(os.path.join(path, 'img_{}.png' .format(epoch)), img_grid)
                
                artificial_image = sess.run(model.x_hat, feed_dict={model.z: np.random.normal(0, 1, (144, model.z_size))})
                
                img_grid = merge(artificial_image[0:144], [12,12])
                img_grid = np.array(img_grid * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(path, 'art_img_{}.png' .format(epoch)), img_grid)
            
                save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                print "Model saved in path: %s" % save_path            
            
            
            print "Step: {}\tValLoss:{}\tTrainLoss: {}\tReconLoss: {}\tLatentLoss: {}" .format( global_step, curr_loss, np.mean(t_loss), np.mean(r_loss), np.mean(l_loss))  

            print "End of epoch {} (took {} minutes)." .format(epoch, round((time.time() - epoch_time)/60, 2)) 
            
            
            
def train_gan(gan, train_dsl, val_dsl, noise_val_dsl, logdir, logdir_true):        
    num_batches_tr  = train_dsl.get_num_batches()
    
    num_samples_val = val_dsl.get_num_samples()
    
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(gan.get_graph()) 
    
    print 'num training batches {}\n'  .format(num_batches_tr)
    print 'GAN train epochs: {} disc epochs: {}' .format(gan.train_epochs, gan.disc_epochs)
    
    orig_var_list = [v for v in tf.compat.v1.global_variables() if v.name.startswith('true_model')]
    orig_saver    = tf.compat.v1.train.Saver(var_list=orig_var_list)
    
    saver    = tf.compat.v1.train.Saver(max_to_keep=300)
    
    path     = logdir + '/images'
    create_dirs([path])
    
    with tf.compat.v1.Session(config =config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
        
        
        z2 = np.random.uniform(-1.0,1.0, size=[cfg.batch_size, cfg.z_size]).astype(np.float32) #Generate another z batch
        
        for epoch in range(1, gan.train_epochs+ gan.disc_epochs + 1):
            epoch_time = time.time()
            dloss_ep = []
            gloss_ep = []
            
            for b_tr in range(num_batches_tr): 

                xs, _ = train_dsl.load_next_batch(b_tr)
                            
                zs = np.random.uniform(-1.0,1.0,size=[xs.shape[0],cfg.z_size]).astype(np.float32)
                _, dLoss = sess.run([gan.d_opt, gan.d_loss],{gan.z_in:zs, gan.X:xs, gan.dropout_keep_prob:0.6})
                dloss_ep.append(dLoss)
                
                if epoch <= gan.train_epochs:                
                    _, gLoss = sess.run([gan.g_opt, gan.g_loss],feed_dict={gan.z_in:zs, gan.dropout_keep_prob:0.6})
                    _, gLoss = sess.run([gan.g_opt, gan.g_loss],feed_dict={gan.z_in:zs, gan.dropout_keep_prob:0.6})
                else:
                    gLoss = sess.run(gan.g_loss,feed_dict={gan.z_in:zs, gan.dropout_keep_prob:1.0})
                
                gloss_ep.append(gLoss)
                
                if math.isnan(dLoss) or math.isnan(gLoss):
                    print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
                    save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                    print "Model saved in path: %s" % save_path                                        
                    raise Exception("Some issue with training DCGAN")
                
                if b_tr % (num_batches_tr/3) == 0:
                    print "Gen Loss: " + str(gLoss) + " Disc Loss: " + str(dLoss)
                
                if epoch % 10 == 0 and b_tr == (num_batches_tr - 1):                    
                    
                    newZ = sess.run(gan.Gz, feed_dict={gan.z_in:z2}) #Use new z to get sample images from generator.
                                        
                    img_grid = merge(newZ[0:144], [12,12])
                    img_grid = np.array(img_grid * 255).astype(np.uint8)                                        
                    imageio.imwrite(os.path.join(path, 'img_{}_{}.png' .format(epoch, 1)), img_grid)
                    
                    z22 = np.random.uniform(-1.0,1.0, size=[cfg.batch_size, cfg.z_size]).astype(np.float32) 
                    
                    newZ = sess.run(gan.Gz, feed_dict={gan.z_in:z22}) #Use new z to get sample images from generator.       
                    img_grid = merge(newZ[0:144], [12,12])
                    img_grid = np.array(img_grid * 255).astype(np.uint8)                                        
                    imageio.imwrite(os.path.join(path, 'img_{}_{}.png' .format(epoch, 2)), img_grid)

                    
            
            correct_preds     = get_metric(gan, sess, val_dsl, gan.sum_real_correct_prediction)
            real_acc_true_val = np.sum(correct_preds)/float(num_samples_val)
            
            correct_preds       = get_metric(gan, sess, noise_val_dsl, gan.sum_real_correct_prediction)
            real_acc_nosie_val  = np.sum(correct_preds)/float(noise_val_dsl.get_num_samples())            
            
            print "Epoch: {}\tGenLoss: {}\tDiscLoss: {}\t DiscRealAccVal: {} DiscRealAccNoise: {}" .format(epoch, np.mean(gloss_ep), np.mean(dloss_ep), real_acc_true_val, real_acc_nosie_val)                          
            if epoch % 10 == 0:            
                save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                print "Model saved in path: %s" % save_path
            
            print "End of epoch {} (took {} minutes).\n" .format(epoch, round((time.time() - epoch_time)/60, 2)) 

        
        
def train_language_model(model, train_dsl, val_dsl, logdir, logdir_true=None):
    num_batches_tr  = train_dsl.get_num_batches()
    num_batches_val = val_dsl.get_num_batches()
    num_samples_val = val_dsl.get_num_samples()
    
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(model.get_graph()) 

    if logdir_true is not None:
        orig_var_list = [v for v in tf.compat.v1.global_variables() if not v.name.startswith(model.var_prefix)]
        orig_saver    = tf.compat.v1.train.Saver(var_list=orig_var_list)
        
    saver         = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session(config =config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        if logdir_true is not None:
            orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))        
        
        curr_pp = None
        best_pp = None
        no_improvement = 0    

        for epoch in range(1, model.train_epochs+1):
            epoch_time = time.time()
            t_loss = []
            for b_tr in range(num_batches_tr): 

                trX, trY, trlengths = train_dsl.load_next_batch(b_tr, return_lengths=True)
                trlengths           = trlengths - 1
                trX_shifted         = trX[:,1:]
                trX_shifted         = append_zero_column(trX_shifted, dtype=np.int32)

                global_step, _, summary_str, loss = sess.run([
                                                 model.global_step,
                                                 model.train_op,
                                                 model.train_summary,
                                                 model.mean_loss
                                              ],
                                              feed_dict={
                                                  model.X       : trX,
                                                  model.lengths : trlengths,
                                                  model.targets : trX_shifted,
                                                  model.dropout_keep_prob: cfg.lm_dropout_keep_prob
                                              })
                t_loss.append(loss)    

            likelihood = []
            
            vlengths   = []
            
            for b_val in range(num_batches_val): 
                valX, valY, vallengths = val_dsl.load_next_batch(b_val, return_lengths=True)                        
                vallengths   = vallengths - 1
                vlengths.append(vallengths)
                valX_shifted = valX[:,1:]
                valX_shifted = append_zero_column(valX_shifted, dtype=np.int32)

                seq_score = sess.run(model.sequence_log_prob_score, { model.X: valX, model.lengths:vallengths, model.targets: valX_shifted    } )
                likelihood.append(seq_score)
                
            vlengths = np.concatenate(vlengths, axis=0)

            likelihood = np.concatenate(likelihood, axis=0)

            total_words = np.sum(vlengths)
            
            normalized_log_prob = np.sum(likelihood)/float(total_words)
            curr_pp             = np.exp(-normalized_log_prob)
            
            print "Epoch: {} TrainLoss: {} Perplexity: {} MeanValScore: {} MaxScore: {} MinScore: {}" .format(epoch, np.mean(t_loss), curr_pp, np.mean(likelihood), np.max(likelihood), np.min(likelihood))  
            
            
            if best_pp is None or curr_pp < best_pp :
                best_pp = curr_pp
                save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                print "Model saved in path: %s" % save_path
                print "[BEST]",
                no_improvement = 0
            else:
                no_improvement += 1

                if (no_improvement % cfg.lm_early_stop_tolerance) == 0:
                    break        

            print "End of epoch {} (took {} minutes)." .format(epoch, round((time.time() - epoch_time)/60, 2)) 
    
    
    
def train_jacobian_substitute(sess, xadv, copy_model, X, Y, jtype, lmbda, lmbda_val):
    X_aug = []
    
    feed_dict = {}
    
    if 'jsma' in jtype:
        feed_dict.update({lmbda: lmbda_val })
    
    for start in range(0, len(X), copy_model.get_batch_size()):
        X_in = X[start:start+copy_model.get_batch_size()]
        Y_in = Y[start:start+copy_model.get_batch_size()]

        feed_dict.update({copy_model.X: X_in })

        if 'color' in jtype:
            print 'in color!!'
            lmbda_val_b = np.random.choice([lmbda_val,-lmbda_val], len(X_in))
            lmbda_val_b = lmbda_val_b.reshape(-1,1,1,1)
            feed_dict.update({lmbda: lmbda_val_b }) 
        elif 't-rnd' in jtype:
            print 'Generating random targets'
            target_val  = np.random.choice(copy_model.get_num_classes(), len(X_in))
            target_val  = one_hot_labels(target_val, copy_model.get_num_classes())
            feed_dict.update({copy_model.labels: target_val }) 
        else:                          
            print 'Unatrgeted version'
            feed_dict.update({copy_model.labels: Y_in })                         # papernot, n_fgsm, n_i-fgsm

        xadv_val = sess.run(xadv, feed_dict)

        X_aug.append(xadv_val)

    X_aug = np.concatenate(X_aug, axis=0)

    return X_aug


def setup_jbda(copy_model, jtype, eps=0.25, iter_epochs = 10):
    print "Creating xadv for synthetic jbda type {}" .format(jtype)
    
    lmbda = None
            
    if 'jsma' in jtype:
        print "Instantiating JSMA attack with lmbda {}" .format(eps)
        with copy_model.get_graph().as_default():
            lmbda      = tf.placeholder_with_default(eps, shape=())
            target     = tf.reduce_sum(tf.multiply(copy_model.prob, copy_model.labels), axis=1)        
            jacobian   = tf.gradients(target, copy_model.X)[0]        
            xadv       = tf.add(copy_model.X, tf.multiply(lmbda, tf.sign(jacobian)))
            xadv       = tf.clip_by_value(xadv, 0.0, 1.0)
    elif 'color' in jtype:
        print "Instantiating color attack with value -{}" .format(eps)
        with copy_model.get_graph().as_default():
            lmbda      = tf.placeholder(tf.float32, shape=(None,1,1,1), name='lmbda')
            xadv       = tf.add(copy_model.X, lmbda)
            xadv       = tf.clip_by_value(xadv, 0.0, 1.0)
    elif jtype=='n-fgsm':
        print "Instantiating untargeted fgsm-{}" .format(eps)
        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps, perturbation_multiplier=1)
    elif jtype=='n-ifgsm':
        print "Instantiating iterative untargeted fgsm-{} with {} epochs" .format(eps, iter_epochs)
        with tf.variable_scope("copy_model", reuse=True):
            xadv = pgd(copy_model, y=copy_model.labels, eps=eps, epochs=iter_epochs, perturbation_multiplier=1)  
    elif jtype=='t-rnd-fgsm':
        print "Instantiating targeted fgsm-{}" .format(eps)
        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps, perturbation_multiplier=-1)
    elif jtype=='t-rnd-ifgsm':
        print "Instantiating iterative targeted fgsm-{} with {} epochs" .format(eps, iter_epochs)
        with tf.variable_scope("copy_model", reuse=True):
            xadv = pgd(copy_model, y=copy_model.labels, eps=eps, epochs=iter_epochs, perturbation_multiplier=-1)  
    else:
        raise Exception("JbDA {} technique unrecognized" .format(jtype))
    
    return xadv, lmbda

        
def train_jacobian(true_model, copy_model, X, Y, dsl, logdir_true, logdir_copy, jtype='jsma', epochs=10, query_budget=10000, shuffle=False, ignore_vars=[], eps=0.25):
    
    
    create_dirs([logdir_copy])
    
    print 'Num of test samples', dsl.get_num_samples()
    
    saver = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints)
    
    xadv, lmbda =  setup_jbda(copy_model, jtype, eps)
    
    orig_var_list  = [v for v in tf.compat.v1.global_variables() if not v.name.startswith(copy_model.var_prefix)]
    
    if len(ignore_vars) > 0:
        orig_var_list = [v for v in orig_var_list if not any([s in v.name for s in ignore_vars]) ]
         
    orig_restorer  = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    init = len(X)
        
    curr_budget = init

    iterations = 1

    while True:        
        if curr_budget >= query_budget:
            break
        else:
            curr_budget += curr_budget 
            iterations += 1
        
    print "Initial Seed: {} Query Budget: {} NumIterations: {}" .format(init, query_budget, iterations)
        
    with tf.compat.v1.Session(config=config) as sess:

        c = 0
                
        for it in range(iterations):
            sess.run(tf.compat.v1.global_variables_initializer())
            
            orig_restorer.restore(sess, tf.train.latest_checkpoint(logdir_true))
            
            curr_acc = compute_evaluation_measure(true_model, sess, dsl, true_model.sum_correct_prediction)
            
            print "\n",
            
            print "Secret model Test acc: ", curr_acc                                
            print "Processing iteration ", it+1
            
            assert(shuffle==False), "Deprecated to maintain the order in which the queries are made"
            
            if shuffle:
                print "Shuffling data" 
                X, Y = shuffle_data(X, Y)
            
            print "Training using {} {}" .format(X.shape, Y.shape)
            
            unique, counts = np.unique(Y, return_counts=True)
            
            print dict(zip(unique, counts))
            
            Y_onehot = one_hot_labels(Y, copy_model.get_num_classes())
                        
            curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
            
            print "Copy model Test Acc before iter {} is {}" .format(it+1,  curr_acc)            
            
            for epoch in range(epochs):
                t_loss = []
                for start in range(0, len(X), copy_model.get_batch_size()):
                    global_step, _, summary_str, loss = sess.run([
                                                                     copy_model.global_step,
                                                                     copy_model.train_op,
                                                                     copy_model.train_summary,
                                                                     copy_model.mean_loss
                                                                  ],
                                                                  {
                                                                    copy_model.X: X[start:start+copy_model.get_batch_size()],
                                                                    copy_model.labels: Y_onehot[start:start+copy_model.get_batch_size()],
                                                                    copy_model.dropout_keep_prob: cfg.dropout_keep_prob,
                                                                  })
                    
                                                            
                    t_loss.append(loss)    
                
                curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class= (not cfg.ignore_invalid_class))
        
                print "Epoch: {} Step: {}  MeanLoss: {} TestAcc: {}" .format(epoch, global_step, np.mean(t_loss), curr_acc)  
                
                if it==(iterations-1) and epoch == (epochs-1):
                    save_path = saver.save(sess, logdir_copy + '/model_epoch_%d' % (epoch))
                    print "Model saved in path: %s" % save_path
            
            curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
            
            print "Copy model Test Acc after iter {} is {}" .format(it+1,  curr_acc)

            secret_preds = get_labels(true_model, sess, dsl, return_true_labels=False)

            copy_preds   = get_labels(copy_model, sess, dsl, return_true_labels=False)

            print "Test agreement between source and copy model on true test dataset", np.sum(secret_preds==copy_preds)/float(len(secret_preds))                    
            
            if it!= iterations -1 :
                
                lmbda_val = eps
                                                
                if 'jsma' in jtype:
                    lmbda_coef = 2 * int(int(it/int(iterations/2)) != 0) - 1      # 2 * int(int(rho / 3) != 0) - 1  
                    lmbda_val  = lmbda_coef * eps            
                    print 'iteration: {} lmbda_val: {}' .format(it+1,  lmbda_val)
                else:
                    print 'iteration: {}' .format(it+1)
                    
                X_aug = train_jacobian_substitute(sess, xadv, copy_model, X, Y_onehot, jtype, lmbda, lmbda_val)

                print 'Samples to Generate ', len(X_aug)        
                    
                Y_aug = get_metric_batch(true_model, sess, X_aug, true_model.predictions)
                
                                                    
                X = np.concatenate([X,X_aug])                            
                Y = np.concatenate([Y,Y_aug])                            

                #Truncate to avoid crossing query budget
                X = X[:query_budget]
                Y = Y[:query_budget]                
                                
                c = c + len(Y_aug)
                
        assert (len(X) == query_budget), 'Budget not completed'
        
        print "Total number of exmaples on which substitute model was trained including seed samples ", len(X)                   
        print X.shape, Y.shape
                                        
        path = os.path.join('prada_results', str(cfg.true_dataset), 'jbda', '{}-{}-{}' .format(jtype, eps, str(query_budget)), 'undefended')
        
        create_dirs([path])
        
        np.save(os.path.join(path, 'X.npy') , X)                
        np.save(os.path.join(path, 'Y.npy') , Y)
        
        
def _line_search(X, Y, idx1, idx2, predict_func, eps, append=False):
    v1 = X[idx1, :]
    y1 = Y[idx1]
    v2 = X[idx2, :]
    y2 = Y[idx2]

    assert np.all(y1 != y2)

    if append:
        samples = X

    # process all points in parallel
    while np.any(np.sum((v1 - v2)**2, axis=-1)**(1./2) > eps):
        # find all mid points
        mid = 0.5 * (v1 + v2)

        # query the class on the current model
        y_mid = predict_func(mid)

        # change either v1 or v2 depending on the value of y_mid
        index1 = np.where(y_mid != y1)[0]
        index2 = np.where(y_mid == y1)[0]

        for idx in np.where(np.logical_and(y_mid != y1, y_mid != y2))[0]:
            pass

        if len(index1):
            v2[index1, :] = mid[index1, :]
        if len(index2):
            v1[index2, :] = mid[index2, :]

        if append:
            samples = np.vstack((samples, mid))

    if append:
        return samples
    else:
        return np.vstack((v1, v2))        
        
        
def all_pairs(Y):
    classes = pd.Series(Y).unique().tolist()

    result = [(i, j)
            for i in range(len(Y))
            for c in classes
            if c != Y[i]
            for j in np.where(Y == c)[0][0:1]
            if i > j]

    return result


def train_copynet_tramer(true_model, copy_model, test_dsl, logdir_true, logdir_copy):
    
    sample_shape = test_dsl.get_sample_shape()
    
    if cfg.sampling_method == "linesearch":
        iterations = 1
    elif cfg.sampling_method == 'adaptive':        
        iterations = 10
    
    if cfg.sampling_method == "linesearch" or cfg.sampling_method == 'adaptive': 
        
        print "Running line search procedure using {} strategy" .format(cfg.sampling_method) 
        sample_limit = int(0.25 * cfg.query_budget)
        
        unif_dsl = UniformDSL(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=False, seed=cfg.seed, sample_limit=sample_limit, shape=sample_shape)
        
        print "LS: Seed size ", len(unif_dsl.data)

        def predict_func(x):
            return simple_predict(true_model, x, logdir_true)

        train_dsl = copy.deepcopy(unif_dsl)
        print "Samples so far: " , len(train_dsl.data)
        
        for iteration in range(iterations):
            Y = predict(true_model, train_dsl, logdir_true, return_true_labels=False)
            print "Class distribution of seed samples ", dict(zip(*np.unique(Y, return_counts=True)))

            idx1, idx2 = zip(*all_pairs(Y))
            idx1 = list(idx1)
            idx2 = list(idx2)
            
            ls_samples = _line_search(train_dsl.data, Y, idx1, idx2, predict_func, 1e-1, append=True)

            print "Number of seed+line search samples generated: " , len(ls_samples)
            
            limit = int(0.25 * cfg.query_budget + 0.75 * cfg.query_budget * ((iteration+1)/iterations))
            
            train_dsl.data = ls_samples[:limit]
            
            print "Samples so far: " , len(train_dsl.data)

        train_dsl.data = ls_samples[:cfg.query_budget]
        print "Final number of samples: " , len(train_dsl.data)
        
        assert(len(train_dsl.data) == cfg.query_budget)
        
    elif cfg.sampling_method == "random":
        print "Randomly generating queries"
        train_dsl = UniformDSL(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=False, seed=cfg.seed, sample_limit=cfg.query_budget, shape=sample_shape)        
    else:
        raise Exception("Method uncrecognized! ", cfg.sampling_method)
    
    train_dsl.labels = predict(true_model, train_dsl, logdir_true, return_true_labels=False)
    print(train_dsl.labels[::cfg.batch_size][:20])
    
    print "Class distribution for training: ", dict(zip(*np.unique(train_dsl.labels, return_counts=True)))
    
    train_dsl.num_classes = test_dsl.get_num_classes()
        
    path = os.path.join('prada_results', cfg.true_dataset, 'tramer', '{}-{}' .format(cfg.sampling_method, cfg.query_budget), 'undefended')
            
    create_dirs(path)
    create_dirs(logdir_copy)
    
    save = np.save(os.path.join(path, 'X.npy') , train_dsl.data)                
    save = np.save(os.path.join(path, 'Y.npy') , train_dsl.labels)
    
    saver = tf.compat.v1.train.Saver()

    print "Training copynet on X: {} Y: {} " .format(train_dsl.data.shape, train_dsl.labels.shape)
    
    orig_var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(true_model.var_prefix)]
             
    orig_restorer  = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    num_batches_tr = train_dsl.get_num_batches()
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
            
        orig_restorer.restore(sess, tf.train.latest_checkpoint(logdir_true))
                                        
        curr_acc = compute_evaluation_measure(true_model, sess, test_dsl, true_model.sum_correct_prediction)            
        print "Secret model Test acc: ", curr_acc

        for epoch in range(50):
            t_loss = []

            for b_tr in range(num_batches_tr):
                trX, trY = train_dsl.load_next_batch(b_tr)    
                global_step, _, summary_str, loss = sess.run([ copy_model.global_step, copy_model.train_op, 
                                                               copy_model.train_summary, copy_model.mean_loss
                                                              ],
                                                              { copy_model.X: trX, copy_model.labels: trY,
                                                                copy_model.dropout_keep_prob: cfg.dropout_keep_prob,
                                                              })


                t_loss.append(loss)

            print "Epoch: {} Step: {}  MeanLoss: {}" .format(epoch, global_step, np.mean(t_loss))  
                
        save_path = saver.save(sess, logdir_copy + '/model_epoch_%d' % (epoch))
        print "Model saved in path: %s" % save_path

        copy_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class= (not cfg.ignore_invalid_class))                

        secret_preds = get_labels(true_model, sess, test_dsl, return_true_labels=False)
        copy_preds   = get_labels(copy_model, sess, test_dsl, return_true_labels=False)

        print "Test agreement between source and copy model on true test dataset", np.sum(secret_preds==copy_preds)/float(len(secret_preds))  

        print "Copy model Test Acc: ", copy_acc

def train_copynet_tramer_svm( true_model, filter_model, svm, copy_model, test_dsl, logdir_true, logdir_copy, ignore_vars=[]):
    
    sample_shape = test_dsl.get_sample_shape()
    
    if cfg.sampling_method == "linesearch":
        iterations = 1
    elif cfg.sampling_method == 'adaptive':        
        iterations = 10
    
    if cfg.sampling_method == "linesearch" or cfg.sampling_method == 'adaptive': 
        
        print "Running line search procedure using {} strategy" .format(cfg.sampling_method) 
        sample_limit = int(0.25 * cfg.query_budget)
        
        unif_dsl = UniformDSL(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=False, seed=cfg.seed, sample_limit=sample_limit, shape=sample_shape)
        
        print "LS: Seed size ", len(unif_dsl.data)

        def predict_func(x):
            return simple_predict(true_model, x, logdir_true)

        train_dsl = copy.deepcopy(unif_dsl)
        print "Samples so far: " , len(train_dsl.data)

        for iteration in range(iterations):
            Y = predict(true_model, train_dsl, logdir_true, return_true_labels=False)
            print "Class distribution of seed samples ", dict(zip(*np.unique(Y, return_counts=True)))

            idx1, idx2 = zip(*all_pairs(Y))
            idx1 = list(idx1)
            idx2 = list(idx2)
            
            ls_samples = _line_search(train_dsl.data, Y, idx1, idx2, predict_func, 1e-1, append=True)

            print "Number of seed+line search samples generated: " , len(ls_samples)
            
            limit = int(0.25 * cfg.query_budget + 0.75 * cfg.query_budget * ((iteration+1)/iterations))
            
            train_dsl.data = ls_samples[:limit]
            
            print "Samples so far: " , len(train_dsl.data)

        train_dsl.data = ls_samples[:cfg.query_budget]
        print "Final number of samples: " , len(train_dsl.data)
        
        assert(len(train_dsl.data) == cfg.query_budget)
        
    elif cfg.sampling_method == "random":
        print "Randomly generating queries"
        train_dsl = UniformDSL(batch_size = cfg.batch_size, mode='train', shuffle_each_epoch=False, seed=cfg.seed, sample_limit=cfg.query_budget, shape=sample_shape)        
    else:
        raise Exception("Method uncrecognized! ", cfg.sampling_method)
    
    train_dsl.labels = predict(true_model, train_dsl, logdir_true, return_true_labels=False)
    
    print "Class distribution for training: ", dict(zip(*np.unique(train_dsl.labels, return_counts=True)))
    
    train_dsl.num_classes = test_dsl.get_num_classes()
        
    path = os.path.join('prada_results', cfg.true_dataset, 'tramer', '{}-{}' .format(cfg.sampling_method, cfg.query_budget), 'undefended')
            
    create_dirs(path)
    create_dirs(logdir_copy)
    
    save = np.save(os.path.join(path, 'X.npy') , train_dsl.data)                
    save = np.save(os.path.join(path, 'Y.npy') , train_dsl.labels)
    
    saver = tf.compat.v1.train.Saver()

    print "Training copynet on X: {} Y: {} " .format(train_dsl.data.shape, train_dsl.labels.shape)
    
    orig_var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(true_model.var_prefix)]
             
    orig_restorer  = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    num_batches_tr = train_dsl.get_num_batches()

    # New edit
    orig_var_list = [v for v in tf.compat.v1.global_variables() if not v.name.startswith( copy_model.var_prefix)]    
    
    if len(ignore_vars) > 0:
        orig_var_list = [v for v in orig_var_list if not any([s in v.name for s in ignore_vars]) ]
            
    orig_saver    = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)    

    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
    
    X = train_dsl.data
    Y = train_dsl.labels

    zs = get_metric_batch(filter_model, sess, X, filter_model.mean) #filter_model.z)
    
    svm_preds = svm.predict(zs)

    print 'svm_preds:'
    print svm_preds[:110]

    assert -1 not in svm_preds[:100]
    assert svm_preds[100] == -1
    
    filtered_indexes = np.where(svm_preds==1)[0]
    print 'filtered_indices:'
    print filtered_indexes

    print 'data before:', train_dsl.data.shape
    train_dsl.data = X[filtered_indexes]
    print 'data after:', train_dsl.data.shape
    # New edit
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
            
        orig_restorer.restore(sess, tf.train.latest_checkpoint(logdir_true))
                                        
        curr_acc = compute_evaluation_measure(true_model, sess, test_dsl, true_model.sum_correct_prediction)            
        print "Secret model Test acc: ", curr_acc

        for epoch in range(50):
            t_loss = []

            for b_tr in range(num_batches_tr):
                trX, trY = train_dsl.load_next_batch(b_tr)    
                global_step, _, summary_str, loss = sess.run(
                    [
                        copy_model.global_step, copy_model.train_op, 
                        copy_model.train_summary, copy_model.mean_loss
                    ],
                    {
                        copy_model.X: trX,
                        copy_model.labels: trY,
                        copy_model.dropout_keep_prob: cfg.dropout_keep_prob,
                    }
                )


            t_loss.append(loss)    

            print "Epoch: {} Step: {}  MeanLoss: {}" .format(epoch, global_step, np.mean(t_loss))  
                
        save_path = saver.save(sess, logdir_copy + '/model_epoch_%d' % (epoch))
        print "Model saved in path: %s" % save_path

        copy_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class= (not cfg.ignore_invalid_class))                

        secret_preds = get_labels(true_model, sess, test_dsl, return_true_labels=False)
        copy_preds   = get_labels(copy_model, sess, test_dsl, return_true_labels=False)

        print "Test agreement between source and copy model on true test dataset", np.sum(secret_preds==copy_preds)/float(len(secret_preds))  
        print "Copy model Test Acc: ", copy_acc

def train_jacobian_svm(secret_model, filter_model, svm, copy_model, X, Y, dsl, logdir_true, logdir_copy, jtype='jsma', epochs=100, query_budget=10000, shuffle=False, ignore_vars=[], eps=0.25):
    
    print 'Num of test samples', dsl.get_num_samples()
    
    saver = tf.compat.v1.train.Saver()
    xadv, lmbda =  setup_jbda(copy_model, jtype, eps)
               
    orig_var_list  = [v for v in tf.compat.v1.global_variables() if not v.name.startswith(copy_model.var_prefix)]
    
    if len(ignore_vars) > 0:
        orig_var_list = [v for v in orig_var_list if not any([s in v.name for s in ignore_vars]) ]
         
    orig_restorer  = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)
    
    init = len(X)
    curr_budget = init
    iterations = 1
    create_dirs(logdir_copy)
    
    while True:        
        if curr_budget >= query_budget:
            break
        else:
            curr_budget += curr_budget 
            iterations += 1
        
    print "Initial Seed: {} Query Budget: {} NumIterations: {}" .format(init, query_budget, iterations)
            
    with tf.compat.v1.Session(config=config) as sess:
        c = 0
        svm_preds = []

        sess.run(tf.compat.v1.global_variables_initializer())
        orig_restorer.restore(sess, tf.train.latest_checkpoint(logdir_true))
        curr_acc = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)

        # run the pure queries through SVM
        zs    = get_metric_batch(filter_model, sess, X, filter_model.mean) #filter_model.z)

        # should not trigger detection    
        svm_preds = svm.predict(zs, skip_check=True)
        assert -1 not in svm_preds
        
        for it in range(iterations):
            if it > 0:
                sess.run(tf.compat.v1.global_variables_initializer())            
                orig_restorer.restore(sess, tf.train.latest_checkpoint(logdir_true))
                curr_acc = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)
            
            print "\n"            
            print "Secret model Test Acc: ", curr_acc            
            #print "Defended model Test Acc: ", compute_svm_defended_model_acc(sess, secret_model, filter_model, svm , dsl)                              
            print "Processing iteration ", it+1
            
            assert(shuffle==False), "Deprecated to maintain the order in which the queries are made"
            
            if shuffle:
                print "Shuffling data"
                X, Y = shuffle_data(X, Y)
            
            print "Training using {} {}" .format(X.shape, Y.shape)
            
            unique, counts = np.unique(Y, return_counts=True)            
            print dict(zip(unique, counts))
            
            Y_onehot = one_hot_labels(Y, copy_model.get_num_classes())
                
            curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction)
            
            print "Copy model Test Acc before iter {} is {}" .format(it+1,  curr_acc)            

            for epoch in range(epochs):
                t_loss = []
                copy_preds = []                
                for start in range(0, len(X), copy_model.get_batch_size()):
                    global_step, _, summary_str, loss = sess.run([   copy_model.global_step,
                                                                     copy_model.train_op,
                                                                     copy_model.train_summary,
                                                                     copy_model.mean_loss
                                                                  ],
                                                                  {
                                                                    copy_model.X: X[start:start+copy_model.get_batch_size()],
                                                                    copy_model.labels: Y_onehot[start:start+copy_model.get_batch_size()],
                                                                    copy_model.dropout_keep_prob: cfg.dropout_keep_prob,
                                                                  })
                    
                    t_loss.append(loss)
                
                curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction)
        
                print "Epoch: {} Step: {}  MeanLoss: {} TestAcc: {}" .format(epoch, global_step, np.mean(t_loss), curr_acc)  
                
                if it==(iterations-1) and epoch == (epochs-1):
                    save_path = saver.save(sess, logdir_copy + '/model_epoch_%d' % (epoch))
                    print "Model saved in path: %s" % save_path
            
            curr_acc = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction)
            
            print "Copy model Test Acc after iter {} is {}" .format(it+1,  curr_acc)
            
            secret_preds = get_labels(secret_model, sess, dsl, return_true_labels=False)            
            copy_preds   = get_labels(copy_model, sess, dsl, return_true_labels=False)            
            
            print "Test agreement between source and copy model on true test dataset", np.sum(secret_preds==copy_preds)/float(len(secret_preds))
                                                    
            if it!= iterations -1 :
                lmbda_val = eps                

                print 'svm_preds', svm_preds

                if -1 in svm_preds:
                    print "Cannot proceed copynet training! All the selected samples rejected. Breaking..."
                    save_path = saver.save(sess, logdir_copy + '/model_epoch_%d' % (epochs - 1))
                    print "Model saved in path: %s" % save_path
                    break
                                                
                if 'jsma' in jtype:
                    lmbda_coef = 2 * int(int(it/int(iterations/2)) != 0) - 1      # 2 * int(int(rho / 3) != 0) - 1  
                    lmbda_val  = lmbda_coef * 0.1            
                    print 'iteration: {} lmbda_val: {}' .format(it+1,  lmbda_val)
                else:
                    print 'iteration: {}' .format(it+1)
                    
                X_aug = train_jacobian_substitute(sess, xadv, copy_model, X, Y_onehot, jtype, lmbda, lmbda_val)

                print 'examples generated: ', len(X_aug)        
                    
                Y_aug = get_metric_batch(secret_model, sess, X_aug, secret_model.predictions)
                
                zs    = get_metric_batch(filter_model, sess, X_aug, filter_model.mean) #filter_model.z)
                
                svm_preds = svm.predict(zs)
                
                filtered_indexes = np.where(svm_preds==1)[0]
                
                X_aug = X_aug[filtered_indexes]
                Y_aug = Y_aug[filtered_indexes]
                
                X = np.concatenate([X,X_aug])                            
                Y = np.concatenate([Y,Y_aug])                            
                
                #Truncate to avoid crossing query budget
                X = X[:query_budget]
                Y = Y[:query_budget]                
                
                c = c + len(Y_aug)
        
        
        num_rejected           = query_budget - len(X)
        num_rejected_exc       = (query_budget - init) - (len(X) - init)
        
        print "Number of invalid queries rejected including seed samples {} " .format(num_rejected )
        print "Number of invalid queries rejected excluding seed samples {} " .format(num_rejected_exc)
        
        print "Total number of exmaples on which substitute model was trained including seed samples ", len(X)            
        
        rejected_ratio     = np.round((num_rejected/float(query_budget))*100.0, 2)
        
        rejected_exc_ratio = np.round((num_rejected_exc/float(query_budget-init))*100.0, 2)
        
        print "\nSVM Noise: "
        
        print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" %(rejected_ratio,100-rejected_ratio, num_rejected, len(X)) 
        
        print "Ratio of noise samples rejected " , rejected_ratio
            
        print "Ratio of noise samples excluding seed rejected " , rejected_exc_ratio
                                                    
        path = os.path.join('prada_results', str(cfg.true_dataset), 'jbda', '{}-{}-{}' .format(jtype, eps, str(query_budget)), cfg.defender_type)
        
        
        print "Saving queries in path: ", path
        
        create_dirs([path])
        
        save = np.save(os.path.join(path, 'X.npy') , X)                
        save = np.save(os.path.join(path, 'Y.npy') , Y)
                        
        
        
def train_mlp(model, trX, trY, logdir):
    "Trains the model and saves the best model in logdir"
    
        
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(model.get_graph()) 
    
    trY = one_hot_labels(trY, 2)
    
    print trX.shape, trY.shape, model.get_batch_size()
    
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session(config =config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
                
        for epoch in range(1, 200):
            epoch_time = time.time()
            t_loss = []
            
            perm = np.arange(trX.shape[0]) 
            random.shuffle(perm)
            trX   = trX[perm]    
            trY   = trY[perm]            
            
            for start in range(0, len(trX), model.get_batch_size()): 
                                    
                feed_dict = { model.X: trX[start:start+model.get_batch_size()], model.labels: trY[start:start+model.get_batch_size()], model.dropout_keep_prob: 0.8 }                
                
                global_step, _, summary_str, loss = sess.run([ model.global_step,
                                                               model.train_op,
                                                               model.train_summary,
                                                               model.mean_loss],
                                                               feed_dict
                                                            )
                                
                t_loss.append(loss)    
                
                train_writer.add_summary(summary_str, global_step)
                train_writer.flush()
        
            if epoch % 10 == 0:                
                save_path = saver.save(sess, logdir + '/model_epoch_%d' % (epoch))       
                print "Model saved in path: %s" % save_path
                    
            print "Step: {} \tTrainLoss: {}" .format( global_step, np.mean(t_loss))  

            print "End of epoch {} (took {} minutes)." .format(epoch, round((time.time() - epoch_time)/60, 2)) 
        
        
def get_test_subset(test_dsl, sub_size):         
        X_test, Y_test = test_dsl.data, test_dsl.labels

        sss = BalancingSelectionStrategy(sub_size, Y_test, test_dsl.get_num_classes())

        s = sss.get_subset()

        X_sub = np.array([X_test[e] for e in s])
        Y_sub = np.array([Y_test[e] for e in s])
        
        X_sub, Y_sub = shuffle_data(X_sub, Y_sub)
                
        unused     = set(range(len(Y_test))) - set(s)
        X_test_new = np.array([X_test[e] for e in unused])
        Y_test_new = np.array([Y_test[e] for e in unused])        
        mod_test_dsl = copy.deepcopy(test_dsl)
        mod_test_dsl.data   = X_test_new        
        mod_test_dsl.labels = Y_test_new        
                
        return X_sub, Y_sub, mod_test_dsl
    

def compute_svm_defended_model_acc(sess, secret_model, filter_model, svm, dsl):
    
    secret_preds, true_labels = get_labels(secret_model, sess, dsl, return_true_labels=True)
        
    secret_match = (secret_preds == true_labels)
    
    secret_match_sum = np.sum(secret_match)
        
    zs        = get_metric(filter_model, sess, dsl, filter_model.mean) #filter_model.z)
    
    svm_preds = svm.predict(zs)
    
    defender_match  = np.logical_and(secret_match, (svm_preds==1)) 
    
    defender_match_sum = np.sum(defender_match)
        
    acc = (defender_match_sum/float(dsl.get_num_samples()))
    
    return acc



def compute_svm_defended_model_preds(sess, secret_model, filter_model, svm, dsl):
    
    secret_preds = get_labels(secret_model, sess, dsl)
    
    zs           = get_metric(filter_model, sess, dsl, filter_model.mean) #.z)
    
    svm_preds    = svm.predict(zs)
    
    mask         = (svm_preds==1)
    
    invalid_class_idx = dsl.get_num_classes()
    
    preds = mask*secret_preds + (1-mask) * invalid_class_idx
        
    return preds


def compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, X):
    
    secret_preds = get_metric_batch(secret_model, sess, X, secret_model.predictions)
    
    zs           = get_metric_batch(filter_model, sess, X, filter_model.mean) #.z)
        
    svm_preds    = svm.predict(zs)
    
    mask         = (svm_preds==1)
    
    invalid_class_idx = secret_model.get_num_classes()
    
    preds = (mask*secret_preds) + ((1-mask) * invalid_class_idx)
        
    return preds


        
def evaluate(model, dsl, logdir, checkpoint=None, ignore_vars=[], add_class=False):
        
    num_samples = dsl.get_num_samples()

    if len(ignore_vars) > 0:
        var_list = [v for v in tf.compat.v1.global_variables() if not any([s in v.name for s in ignore_vars]) ]           
        saver = tf.compat.v1.train.Saver(var_list=var_list)
    else:
        saver = tf.compat.v1.train.Saver()
        
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))                                                    
        else:
            saver.restore(sess, checkpoint)
        
        accuracy = compute_evaluation_measure(model, sess, dsl, model.sum_correct_prediction, add_class=add_class)
        
        return accuracy
        

def evaluate_ae(model, dsl, logdir, dae, logdir_ae):
        
    num_samples = dsl.get_num_samples()
    
    orig_var_list = [v for v in tf.compat.v1.global_variables() if not v.name.startswith('copy_model') and not v.name.startswith('dae')]
    
    dae_var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith('dae')]
    
    true_saver = tf.compat.v1.train.Saver(var_list=orig_var_list)
    
    dae_saver  = tf.compat.v1.train.Saver(var_list=dae_var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        true_saver.restore(sess, tf.train.latest_checkpoint(logdir))
        dae_saver.restore(sess, tf.train.latest_checkpoint(logdir_ae))
            
        accuracy = compute_evaluation_measure(model, sess, dsl, model.sum_correct_prediction, dae=dae)
        print "Accuracy:", accuracy



def compute_reconstruction_losses(model, dsl, logdir_ae, metric=None):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    if metric is None:
        metric = model.recon_loss_mse    
    
    with tf.compat.v1.Session(config=config) as sess:        
        print "Loading filter model from dir: " , logdir_ae
        
        saver.restore(sess, tf.train.latest_checkpoint(logdir_ae))
                    
        rlosses = get_metric(model, sess, dsl, metric)
        
        return rlosses
    
    
def get_decoded_losses_imgs(model, dsl, logdir):
    
    rlosses      = []
    decoded_imgs = []
    
    rlosses_mae  = []

    saver = tf.compat.v1.train.Saver([ v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)])
                
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))                    
                
        for b in range(dsl.get_num_batches()):
            X, _ = dsl.load_next_batch(b)
            
            decoded, rloss, rloss_mae = sess.run([model.decoded, model.recon_loss_mse, model.recon_loss_mae] , {model.X: X })
            rlosses.append(rloss)
            decoded_imgs.append(decoded)            
            rlosses_mae.append(rloss_mae)

        decoded_imgs = np.concatenate(decoded_imgs)
        rlosses      = np.concatenate(rlosses)  
    
        rlosses_mae  = np.concatenate(rlosses_mae)  
        
    
    return rlosses, rlosses_mae, decoded_imgs    



def get_decoded_imgs(model, dsl, logdir):
    
    decoded_imgs = []
                
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    saver  = tf.compat.v1.train.Saver(var_list=var_list)        
        
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))                    
                
        for b in range(dsl.get_num_batches()):
            X, _ = dsl.load_next_batch(b)

            decoded = sess.run(model.decoded, {model.X: X })
            decoded_imgs.append(decoded)

        decoded_imgs = np.concatenate(decoded_imgs)
    
    return decoded_imgs    


    
def compute_sequence_log_prob_score(model, dsl, logdir, metric=None):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    if metric is None:
        metric = model.sequence_log_prob_score    
    
    with tf.compat.v1.Session(config=config) as sess:      
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
                    
        seq_scores = get_scores(model, sess, dsl, metric)
        
        return seq_scores    
    

def sample_z(model, dsl, logdir_vae):
        
    num_samples = dsl.get_num_samples()
    
    vae_var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    vae_saver  = tf.compat.v1.train.Saver(var_list=vae_var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        vae_saver.restore(sess, tf.train.latest_checkpoint(logdir_vae))
                    
        zs = get_metric(model, sess, dsl, model.mean) #.z)
        
        return zs

def sample_z_circled(model, dsl, logdir):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
                            
        imgs = get_decoded_imgs(model, dsl, logdir)
                
        circled_dsl = copy.deepcopy(dsl)
        
        circled_dsl.data = imgs
        
        zs   = get_metric(model, sess, circled_dsl, model.mean) #.z)
        
        return zs
    
    
    
def get_prob(model, dsl, logdir):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
                    
        probs = get_metric(model, sess, dsl, model.prob)
        
        return probs    
    
    
def get_recon_ce(model, dsl, logdir):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
                    
        ce = get_metric(model, sess, dsl, model.recon_loss_ce)
        
        assert np.isnan(ce).any() == False, "Contains NaN"
        
        ce = ce.reshape(-1,1)
        
        mse = get_metric(model, sess, dsl, model.recon_loss_mse)
        
        mse = mse.reshape(-1,1)
        
        mae = get_metric(model, sess, dsl, model.recon_loss_mae)
        
        mae = mae.reshape(-1,1)
        
        recon = np.concatenate([ ce, mse, mae], axis=-1)
        
        return recon        
    
    
def get_entropy(model, dsl, logdir):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]
    
    saver  = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(logdir))
        
        entropy = get_metric(model, sess, dsl, model.entropy)
        
        entropy = entropy.reshape(-1,1)
                
        return entropy            

def print_mean_std(_zs, moms_no=5):
    mean_val = np.mean(_zs, axis=0)
    std_val  = np.std(_zs, axis=0)
    
    print np.linalg.norm(mean_val)
    #print std_val
    
    moms = []
    
    for i in range(1, moms_no+1):
        mom = scipy.stats.moment(_zs, axis=0, moment=i)
        moms.append([mom])
        
    return np.concatenate(moms)

tt_mu = None
tt_sigma = None
tt_train_subs = None

def test_mmd(train, val, test, draws=20):
    global tt_mu, tt_sigma, tt_train_subs

    samples_per_draw = len(test)
    val_mmds = []

    print 'samples_per_draw', samples_per_draw
    
    random_state = np.random.get_state()

    if tt_mu == None:
        np.random.seed(150)

        for _ in range(draws):
            train_subs = train[np.random.choice(train.shape[0], samples_per_draw, replace=False)]
            val_subs = val[np.random.choice(val.shape[0], samples_per_draw, replace=False)]
            
            val_mmds.append(compute_mmd(train_subs, val_subs))
            
        tt_mu = np.mean(val_mmds)
        tt_sigma = np.std(val_mmds)
        tt_train_subs = train_subs
        print 'computing and assigning train/val MMD...'
    
    mu = tt_mu
    sigma = tt_sigma
    train_subs = tt_train_subs
    
    res = compute_mmd(train_subs, test)
    accept = (res <= 0.25) #(res <= mu + 6 *sigma)

    np.random.set_state(random_state)

    if not accept:
        print val_mmds
        print 'Target mu:', mu
        print 'Target sigma:', sigma
        print 'Range: < ', mu, '+', 6*sigma
    
        print 'MMD obtained:', res
    
    return accept

def load_svm(model, secret_model, train_dsl, val_dsl, test_dsl, noise_train_dsl=None,  logdir_defender=None, train=False):
    class FakeSVM:
        def __init__(self, model, train_dsl, val_dsl, logdir_defender, buffer_capacity=100):
            self.query_zs = []
            self.true_train_zs = sample_z(model, train_dsl,  logdir_defender)
            self.true_val_zs = sample_z(model, val_dsl,  logdir_defender)
            self.buffer_capacity = buffer_capacity
            self.client_banned = False
            self.model = model
            self.logdir_vae = logdir_defender
            self.processed = 0

        def process_sample(self, sample_z, skip_check=False):
            self.processed += 1

            # early exit if client banned
            if self.client_banned:
                return -1

            if len(self.query_zs) < self.buffer_capacity:
                # while buffer is not full, admit all queries
                self.query_zs.append(sample_z)
                return +1
            
            elif skip_check:
                # if skipping check, admit query without testing
                self.query_zs.pop(0)
                self.query_zs.append(sample_z)
                return +1

            else:
                # delete element from buffer
                self.query_zs.pop(0)
                self.query_zs.append(sample_z)

                print "Query \#{}:".format(self.processed)

                if test_mmd(self.true_train_zs, self.true_val_zs, self.query_zs):
                    return +1
                else:
                    print '[Client banned] After', self.processed - 1, 'samples were allowed'
                    self.client_banned = True
                    return -1

        def predict(self, query_zs, skip_check=False):
            if skip_check:
                assert not self.client_banned

            result = []

            for query_z in query_zs:
                result.append(self.process_sample(query_z, skip_check))

            return np.array(result)

    fake_svm = FakeSVM(model, train_dsl, val_dsl, logdir_defender)
    
    return fake_svm

def load_svm_old(model, secret_model, train_dsl, val_dsl, test_dsl, noise_train_dsl=None,  logdir_defender=None, train=False):
      
    assert(cfg.svm_threshold is None), "Probability based SVM is not supported"
        
    true_train_zs  = sample_z(model, train_dsl,  logdir_defender)

    gaussian_noise_added_train_dsl = copy.deepcopy(train_dsl)

    num_train_samples = train_dsl.get_num_samples() 
                
    print gaussian_noise_added_train_dsl.data.shape

    locs              = np.random.uniform(0, 1, size=num_train_samples)

    noise_generated   = np.concatenate([[np.random.normal(loc=loc, scale=1.0, size=gaussian_noise_added_train_dsl.data.shape[1:]) for loc in locs]])

    vnoise            = np.random.uniform(low=cfg.vnoise_min, high=cfg.vnoise_max,  size=num_train_samples)

    vnoise            = vnoise.reshape(-1,1, 1, 1)    
            
    gaussian_noise_added_train_dsl.data = np.clip((vnoise * noise_generated) + ((1-vnoise) * train_dsl.data), 0.0, 1.0)        
    gaussian_added_zs                   = sample_z(model, gaussian_noise_added_train_dsl,  logdir_defender) 

    train_labels                        = np.ones(train_dsl.get_num_samples()) 
     
    gaussian_added_labels               = np.ones(gaussian_noise_added_train_dsl.get_num_samples()) * -1

    total_data     = np.concatenate([ train_dsl.data, gaussian_noise_added_train_dsl.data ], axis=0)
    total_zs       = np.concatenate([ true_train_zs, gaussian_added_zs  ], axis=0)
    total_labels   = np.concatenate([ train_labels, gaussian_added_labels ], axis=0)

    print total_zs.shape, total_labels.shape    
    
    if train:
        svm = SVC(C=cfg.C, gamma=cfg.gamma)    
        
        print "Training SVM.."
        
        start_time = time.time()

        svm.fit(total_zs, total_labels)

        print "SVM train time: {} min" .format(round((time.time() - start_time)/60, 2)) 

        joblib.dump(svm, os.path.join(logdir_defender, "{}.pkl" .format('svm')) )
    else:
        svm = joblib.load( os.path.join(logdir_defender, "{}.pkl" .format('svm')) )
    
    print "\nSVM Train:"

    print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" % svm_predictor(svm, total_zs, threshold=cfg.svm_threshold)    
    
    print "Train Accuracy", svm.score(total_zs, total_labels)
    
    rejected_ratio, accepted_ratio, num_rejected, num_accepted = svm_predictor(svm, true_train_zs, threshold=cfg.svm_threshold)
    
    print "Ratio of clean-training samples rejected ", np.round(rejected_ratio, 2)
        
    print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" %(rejected_ratio, accepted_ratio, num_rejected, num_accepted)     
        
    rejected_ratio, accepted_ratio, num_rejected, num_accepted = svm_predictor(svm, gaussian_added_zs, threshold=cfg.svm_threshold)
    
    print "Ratio of noisy-training samples rejected " , np.round(rejected_ratio, 2) 
        
    print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" %(rejected_ratio, accepted_ratio, num_rejected, num_accepted)             
            
    print "\nSVM Validation:"
    
    val_zs     = sample_z(model, val_dsl,  logdir_defender)
    
    rejected_ratio, accepted_ratio, num_rejected, num_accepted = svm_predictor(svm, val_zs, threshold=cfg.svm_threshold)
    
    print "Ratio of validation samples rejected " , np.round(rejected_ratio, 2)
        
    print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" %(rejected_ratio, accepted_ratio, num_rejected, num_accepted)    
        
    print "\nTest:"
    
    test_zs = sample_z(model, test_dsl, logdir_defender)
    
    rejected_ratio, accepted_ratio, num_rejected, num_accepted, svm_preds = svm_predictor(svm, test_zs, threshold=cfg.svm_threshold , return_preds=True)
        
    print "Ratio of test samples rejected " , np.round(rejected_ratio, 2)
        
    print "(SVM) Rejected(%): {} Accepted(%): {} NumRejected: {} NumAccepted: {}" .format(rejected_ratio, accepted_ratio, num_rejected, num_accepted)    
     
    secret_preds, true_labels = predict(secret_model, test_dsl, logdir_defender, return_true_labels=True)
        
    secret_match = (secret_preds == true_labels)
    
    secret_match_sum = np.sum(secret_match)
    
    defender_match  = np.logical_and( secret_match, (svm_preds==1)) 
    
    defender_match_sum = np.sum(defender_match)
    
    print "Secret Model Accuracy: {} SecretMatch: {}" .format((secret_match_sum/float(test_dsl.get_num_samples())) *100.0,secret_match_sum) 
    
    print "Defended Model Accuracy: {} DefenderMatch: {}" .format((defender_match_sum/float(test_dsl.get_num_samples())) * 100.0, defender_match_sum)      
    
    if noise_train_dsl is not None:
        noise_zs = sample_z(model, noise_train_dsl,  logdir_defender)
        
        rejected_ratio, accepted_ratio, num_rejected, num_accepted = svm_predictor(svm, noise_zs, threshold=cfg.svm_threshold)
                      
        print "\nSVM Noise: "
        print "Rejected(%%): %s Accepted(%%): %s Num rejected: %s Num accepted: %s" %(rejected_ratio, accepted_ratio, num_rejected, num_accepted)
        
        print "Ratio of noise samples rejected " , np.round(rejected_ratio, 2)
        
    return svm


def svm_predictor(svc, z, threshold=None, neg_class = -1, return_preds=False):
    
    if threshold is None:
        num_rejected = np.sum(svc.predict(z)==neg_class)
    else:
        probs        = svc.predict_proba(z)
        neg_probs    = probs[:,0]        
        num_rejected = np.sum(neg_probs > threshold)
        
    num_samples    = len(z)
    num_accepted   = num_samples - num_rejected
    
    rejected_ratio  = (num_rejected/float(num_samples)) * 100.0 
    
    accepted_ratio  = (num_accepted/float(num_samples)) * 100.0 
    
    preds           = svc.predict(z)
    
    if return_preds:
        return rejected_ratio, accepted_ratio, num_rejected, num_accepted, preds
    else:
        return rejected_ratio, accepted_ratio, num_rejected, num_accepted     
    
        
def predict(model, dsl, logdir, checkpoint=None, return_true_labels=False):
        
    num_samples = dsl.get_num_samples()
    
    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]    
    saver = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            saver.restore(sess, checkpoint)
        
        predictions = []
        true_labels = []
        num_batches = dsl.get_num_batches()
        
        for step in range(num_batches): 
            X, Y = dsl.load_next_batch(step)
            predictions.append(sess.run(model.predictions, feed_dict={ model.X: X }))
            
            if return_true_labels:
                true_labels.append(Y.argmax(-1))
    
    if return_true_labels:
        return np.concatenate(predictions), np.concatenate(true_labels)
    else:
        return np.concatenate(predictions)

def predict_prada(model, dsl, logdir, checkpoint=None, return_true_labels=False):
        
    num_samples = dsl.get_num_samples()
    
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session(config=config) as sess:
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            saver.restore(sess, checkpoint)
        
        complete_X = []
        predictions = []
        true_labels = []
        num_batches = dsl.get_num_batches()
        
        for step in range(num_batches): 
            X, Y = dsl.load_next_batch(step)
            predictions.append(
                sess.run(
                    model.predictions,
                    feed_dict={
                        model.X: X,
                        model.dropout_keep_prob: 1.0
                    }
               )
           )
            complete_X.append(X)
            if return_true_labels:
                true_labels.append(Y.argmax(-1))
    
    if return_true_labels:
        return np.concatenate(predictions), np.concatenate(true_labels), complete_X
    else:
        return np.concatenate(predictions), complete_X
    
def simple_predict(model, x, logdir, checkpoint=None):

    var_list  = [v for v in tf.compat.v1.global_variables() if v.name.startswith(model.var_prefix)]    
    saver = tf.compat.v1.train.Saver(var_list=var_list)
    
    with tf.compat.v1.Session(config=config) as sess:
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            saver.restore(sess, checkpoint)
        
        predictions = []
        
        for start in range(0, len(x), model.get_batch_size()):
            predictions.append(
                sess.run(
                    model.predictions,
                    feed_dict={
                        model.X: x[start:start+model.get_batch_size()],
                        model.dropout_keep_prob: 1.0
                    }
               )
           )
    
    return np.concatenate(predictions)


def get_test_predictions(model, dsl, logdir, checkpoint=None):
    saver = tf.compat.v1.train.Saver()
    
    with tf.compat.v1.Session(config=config) as sess:
        if checkpoint is None:
            saver.restore(sess, tf.train.latest_checkpoint(logdir))
        else:
            saver.restore(sess, checkpoint)
        
        predictions = []
        
        dsl.reset_batch_counter()
    
        for step in range(dsl.get_num_batches()): 
            X, Y = dsl.load_next_batch(step)
        
            predictions.append(sess.run(model.predictions,
                     feed_dict={
                         model.X: X
                     })
           )
    
    return np.concatenate(predictions)


def compute_f1_measure(model, sess, dsl, use_aux=False, average='macro'):
    assert not model.is_multilabel()
    
    total_measure = 0
    num_batches = dsl.get_num_batches()
    num_samples = dsl.get_num_samples()
    num_classes = model.get_num_classes()
    
    preds = []
    trues = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        if not use_aux:
            X, Y = dsl.load_next_batch()
        else:
            X, _, aux = dsl.load_next_batch(return_aux=use_aux)
            Y         = collect_aux_data(aux, 'true_prob')
            
        pred  = sess.run(model.predictions,
                          feed_dict={
                              model.X: X,
                              model.dropout_keep_prob: 1.0
                          }
                       )
        preds.append(pred)
        trues.append(Y)
    
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    trues = np.argmax(trues,axis=-1)
    
    print "Unique predictions made:", np.unique(preds)
    
    return f1_score(y_true=trues, y_pred=preds , average=average)



def append_class(Y):
    
    temp = np.zeros((Y.shape[0], Y.shape[1]+1))
    temp[:,:-1] = Y
    
    return temp 


def compute_evaluation_measure(model, sess, dsl, measure, use_aux=False, add_class=False):
    total_measure = 0
    num_batches = dsl.get_num_batches()
    num_samples = dsl.get_num_samples()
    
    preds = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        if not use_aux:
            X, Y = dsl.load_next_batch()
        else:
            X, _, aux = dsl.load_next_batch(return_aux=use_aux)
            Y         = collect_aux_data(aux, 'true_prob')
                        
        if add_class:
            Y = append_class(Y)
                
        measure_val = sess.run(measure,
                             feed_dict={
                                 model.X: X,
                                 model.labels: Y,
                                 model.dropout_keep_prob: 1.0
                             }
                           )
        
        total_measure += measure_val
    
    if model.is_multilabel():
        num_classes = model.get_num_classes()
        return total_measure/float(num_samples * num_classes)
    else:
        return total_measure/float(num_samples)    


    
def get_metric(model, sess, dsl, metric):
    
    num_batches = dsl.get_num_batches()
    
    preds = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        X, _ = dsl.load_next_batch()
            
        pred  = sess.run(metric, feed_dict={ model.X: X })
        preds.append(pred)
        
    if np.isscalar(preds[0]):
        preds = np.array(preds)
    else:
        preds = np.concatenate(preds)
    
    return preds    


def get_metric_batch(model, sess, X, metric):
    
    preds = []
        
    for start in range(0, len(X), model.get_batch_size()):
        pred  = sess.run(metric, feed_dict={ model.X: X[start:start+model.get_batch_size()] })
        preds.append(pred)
        
    if np.isscalar(preds[0]):
        preds = np.array(preds)
    else:
        preds = np.concatenate(preds)
    
    return preds    



def get_scores(model, sess, dsl, metric):
    
    num_batches = dsl.get_num_batches()
    
    preds = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches):             
        X, Y, lengths = dsl.load_next_batch(step, return_lengths=True)                        
        lengths   = lengths - 1
        X_shifted = X[:,1:]
        X_shifted = append_zero_column(X_shifted, dtype=np.int32)                        
        seq_score = sess.run(model.sequence_log_prob_score, { model.X: X, model.lengths:lengths, model.targets: X_shifted } )
          
        preds.append(seq_score)
        
    if np.isscalar(preds[0]):
        preds = np.array(preds)
    else:
        preds = np.concatenate(preds)
    
    return preds    
    
    
def get_labels(model, sess, dsl, return_true_labels=False):
    
    num_batches = dsl.get_num_batches()
    
    preds       = []
    
    true_labels = []
    
    dsl.reset_batch_counter()
    
    for step in range(num_batches): 
        X, Y = dsl.load_next_batch()
                    
        pred  = sess.run(model.predictions, feed_dict={ model.X: X })
        preds.append(pred)
        
        true_labels.append(np.argmax(Y, axis=-1))
        
    preds = np.concatenate(preds)
    
    true_labels = np.concatenate(true_labels, axis=0)
    
    if return_true_labels:    
        return preds, true_labels
    else:
        return preds

def get_predictions(sess, model, x, one_hot=False, drop_out=1.0, labels=False):
    
    Y      = []
    Y_prob = []
    Y_idx  = []
    
    for start in range(0, len(x), model.get_batch_size()):
        Y_b, Y_prob_b, Y_idx_b  = sess.run(
            [
                model.predictions_one_hot,
                model.prob,
                model.predictions
            ], 
            feed_dict = {
                model.X: x[start:start+model.get_batch_size()],
                model.dropout_keep_prob:drop_out 
            } 
       )
        
        Y.append(Y_b)
        Y_prob.append(Y_prob_b)
        Y_idx.append(Y_idx_b)
    
    Y      = np.concatenate(Y)
    Y_prob = np.concatenate(Y_prob)
    Y_idx  = np.concatenate(Y_idx)
    
    if one_hot:
        if labels:
            return Y, Y_idx
        else:
            return Y
    else:
        if labels:
            return Y_prob, Y_idx
        else:
            return Y_prob


# For KCenter
def get_initial_centers(sess, noise_train_dsl_marked, copy_model):
    Y_vec_true = []

    noise_train_dsl_marked.reset_batch_counter()
    for b in range(noise_train_dsl_marked.get_num_batches()):
        trX, _ = noise_train_dsl_marked.load_next_batch()
        trY    = get_predictions(sess, copy_model, trX, labels=False)
        Y_vec_true.append(trY)

    Y_vec_true  = np.concatenate(Y_vec_true)

    return Y_vec_true

# For KCenter
def true_initial_centers(sess, noise_train_dsl_marked):
    Y_vec_true = []

    noise_train_dsl_marked.reset_batch_counter()
    for b in range(noise_train_dsl_marked.get_num_batches()):
        trX, _, trY_aux  = noise_train_dsl_marked.load_next_batch(return_idx=False, return_aux=True)                        
        trY              = collect_aux_data(trY_aux, 'true_prob')
        Y_vec_true.append(trY)

    Y_vec_true  = np.concatenate(Y_vec_true)

    return Y_vec_true


def generate_adversarial(xadv, secret_model, copy_model,  dsl, logdir):
                
    num_samples        = dsl.get_num_samples()        
    invalid_class_idx  = dsl.get_num_classes()

    print "Test num_samples: {}" .format(num_samples)
    print "num classes: {}" .format(dsl.get_num_classes())
    print "invalid_class_idx: {}" .format(invalid_class_idx)
        
    var_list    = [v for v in tf.compat.v1.global_variables() if not 'cw' in v.name ]
    restorer    = tf.compat.v1.train.Saver(var_list=var_list)
            
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))

        
        tsuccess   = 0
        asuccess   = 0
        
        true_acc  = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
                
        copy_acc  = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        
        print "Copy model Test Acc: {}" .format(copy_acc)            
            
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
            
            clean_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: X })
            
            clean_pred_copy, clean_pred_copy_one_hot = sess.run([copy_model.predictions, copy_model.predictions_one_hot] , feed_dict = { copy_model.X: X })
                        
            xadv_val       = sess.run(xadv , feed_dict = { copy_model.X: X, copy_model.labels : clean_pred_copy_one_hot })
                
            xadv_pred_copy = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
                            
            xadv_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: xadv_val })
            
            tsuccess += np.sum(np.logical_and((xadv_pred_secret!=clean_pred_secret), (xadv_pred_secret!=invalid_class_idx)))                                                                                                   
            asuccess += np.sum(np.logical_and((clean_pred_copy != xadv_pred_copy), (xadv_pred_copy!=invalid_class_idx)))
                
    
        print 'FGSM Untargeted Transferability success: {}' .format(tsuccess/float(num_samples))
               
        print 'Copy model FGSM Untargeted substitute attack success: {}' .format(asuccess/float(num_samples) )
                
            
            
def generate_adversarial_targeted(xadv, secret_model, copy_model,  dsl, logdir, target_class=2):

    num_samples        = dsl.get_num_samples()            
    num_classes        = dsl.get_num_classes()    
    invalid_class_idx  = dsl.get_num_classes()

    print "num classes: {}" .format(num_classes)
    print "invalid_class_idx: {}" .format(invalid_class_idx)
    print "Target class: {} most likely label" .format(target_class)
    
    restorer      = tf.compat.v1.train.Saver()        
            
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))
        
        tsuccess    = 0
        asuccess    = 0
        
        true_acc  = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
                
        copy_acc  = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        
        print "Copy model Test Acc: {}" .format(copy_acc)        
        
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
            
            clean_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: X })
            
            clean_pred_copy, clean_prob_copy = sess.run([copy_model.predictions, copy_model.prob], feed_dict={copy_model.X:X})
                        
            L       = np.argsort(-clean_prob_copy, axis=-1)
            targets = L[ :, target_class-1 ]                                    
            targets_one_hot   = one_hot_labels(targets, num_classes)
                        
            xadv_val       = sess.run(xadv , feed_dict = { copy_model.X: X, copy_model.labels : targets_one_hot })
                
            xadv_pred_copy = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
                            
            xadv_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: xadv_val })
   
            tsuccess += np.sum(np.logical_and(xadv_pred_secret==targets, xadv_pred_secret!=invalid_class_idx))
            
            asuccess += np.sum(np.logical_and(targets == xadv_pred_copy, xadv_pred_copy!=invalid_class_idx))
                
            
        print 'FGSM Targeted-{} Transferability success: {}' .format(target_class, tsuccess/float(num_samples))
        
        print 'Copy model FGSM Targeted-{} substitute attack success: {}' .format(target_class, asuccess/float(num_samples))
                
        
        
                                
        
def generate_adversarial_svm_filtered(xadv, secret_model, filter_model, svm, copy_model, dsl, logdir):
            
    assert (cfg.ignore_invalid_class == True), 'Currently only ignore invalid class supported'
    
    invalid_class_idx  = dsl.get_num_classes()

    print "num classes: {}" .format(dsl.get_num_classes())
    print "invalid_class_idx: {}" .format(invalid_class_idx)
        
    restorer      = tf.compat.v1.train.Saver()        
            
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))

        num_samples   = dsl.get_num_samples()
        valid_classes = dsl.get_num_classes()
        
        tsuccess    = 0
        asuccess    = 0
        
        true_acc     = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
        copy_acc     = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        print "Copy model Test Acc: {}" .format(copy_acc)            
                        
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
                        
            clean_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, X)
            
            clean_pred_copy, clean_pred_copy_one_hot = sess.run([copy_model.predictions, copy_model.predictions_one_hot] , feed_dict = { copy_model.X: X })
            
            xadv_val       = sess.run(xadv , feed_dict = { copy_model.X: X, copy_model.labels : clean_pred_copy_one_hot })
                
            xadv_pred_copy = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
            
            xadv_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, xadv_val)
                                                            
            tsuccess += np.sum(np.logical_and(xadv_pred_secret!=clean_pred_secret, xadv_pred_secret!=invalid_class_idx))   
            
            asuccess += np.sum(np.logical_and(clean_pred_copy != xadv_pred_copy, xadv_pred_copy!=invalid_class_idx))
        
        
        print 'FGSM Untargeted Transferability success: {}' .format(tsuccess/float(num_samples))
        print 'Copy model FGSM Untargeted substitute attack success:{}' .format(asuccess/float(num_samples))
        

def generate_adversarial_svm_filtered_targeted(xadv, secret_model, filter_model, svm, copy_model, dsl, logdir, target_class=2):
            
    assert (cfg.ignore_invalid_class == True), 'Currently only ignore invalid class supported'
    
    num_classes        = dsl.get_num_classes()
    invalid_class_idx  = dsl.get_num_classes()

    print "num classes: {}" .format(num_classes)
    print "invalid_class_idx: {}" .format(invalid_class_idx)
    print "Target class: {} most likely label" .format(target_class)
    
    restorer      = tf.compat.v1.train.Saver()        
            
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))

        num_samples   = dsl.get_num_samples()
        valid_classes = dsl.get_num_classes()
        
        tsuccess    = 0
        asuccess    = 0
        
        true_acc     = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
        copy_acc     = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        print "Copy model Test Acc: {}" .format(copy_acc)            
                        
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
                        
            clean_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, X)
            
            clean_pred_copy, clean_prob_copy = sess.run([copy_model.predictions, copy_model.prob] , feed_dict = { copy_model.X: X })
            
            L                = np.argsort(-clean_prob_copy, axis=-1)
            targets          = L[ :, target_class-1 ]                                    
            targets_one_hot  = one_hot_labels(targets, num_classes)
            
            xadv_val       = sess.run(xadv , feed_dict = { copy_model.X: X, copy_model.labels : targets_one_hot })
                
            xadv_pred_copy = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
            
            xadv_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, xadv_val)
                                                            
            tsuccess += np.sum(np.logical_and(xadv_pred_secret==targets, xadv_pred_secret!=invalid_class_idx))       
            asuccess += np.sum(np.logical_and(targets == xadv_pred_copy, xadv_pred_copy!=invalid_class_idx))
        
        print 'FGSM Targeted-{} Transferability success: {}' .format(target_class, tsuccess/float(num_samples))
        print 'Copy model FGSM Targeted-{} attack success:{}' .format(target_class, asuccess/float(num_samples))
        

def generate_adversarial_cw_targeted(secret_model, copy_model, dsl, logdir, target_class=2, epochs=100):
        
    targets_placeholder = tf.placeholder(tf.int32, shape=(None,), name='targets')

    height, width, channels = dsl.get_sample_shape()
    
    with tf.variable_scope("copy_model",reuse=tf.AUTO_REUSE):
        train_op, xadv, noise = cw(copy_model, y=targets_placeholder, eps=0.01, xshape=(cfg.batch_size, height, width, channels))    
        zero_noise = tf.assign(noise, tf.zeros_like(noise))
                            
    var_list    = [v for v in tf.compat.v1.global_variables() if not 'cw' in v.name and not v.name.startswith('copy_model_1') ]
    
    restorer    = tf.compat.v1.train.Saver(var_list=var_list)
                        
    num_classes        = dsl.get_num_classes()    
    invalid_class_idx  = dsl.get_num_classes()

    print "num classes: {}" .format(num_classes)
    print "invalid_class_idx: {}" .format(invalid_class_idx)
    print "Target class: {} most likely label" .format(target_class)
                   
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))

        num_samples   = dsl.get_num_samples()
        valid_classes = dsl.get_num_classes()
        
        fsuccess    = 0
        psuccess    = 0
        asuccess    = 0
        
        true_acc  = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
        copy_acc  = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        print "Copy model Test Acc: {}" .format(copy_acc)        
        
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
            
            clean_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: X })
            
            clean_pred_copy, clean_prob_copy = sess.run([copy_model.predictions, copy_model.prob], feed_dict={copy_model.X:X})
                        
            L               = np.argsort(-clean_prob_copy, axis=-1)
            targets         = L[ :, target_class-1 ]                                    
                 
            _ = sess.run(zero_noise)

            for _ in range(epochs):
                _, xadv_val = sess.run([train_op, xadv], {copy_model.X: X, targets_placeholder:targets})
                
            xadv_pred_copy   = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
                            
            xadv_pred_secret = sess.run(secret_model.predictions , feed_dict = { secret_model.X: xadv_val })
   
            fsuccess += np.sum(np.logical_and(xadv_pred_secret==targets, xadv_pred_secret!=invalid_class_idx))
            psuccess += np.sum(np.logical_and(xadv_pred_secret!=clean_pred_secret, xadv_pred_secret!=invalid_class_idx))   
            asuccess += np.sum(np.logical_and(targets == xadv_pred_copy, xadv_pred_copy!=invalid_class_idx))
                
        print 'CW Targeted-{} Partial Transferability success: {}' .format(target_class, psuccess/float(num_samples))
        print 'CW Targeted-{} Full Transferability success: {}' .format(target_class, fsuccess/float(num_samples))
        print 'Copy model CW Targeted-{} attack success: {}' .format(target_class, asuccess/float(num_samples))
        
        
def generate_adversarial_cw_svm_filtered_targeted(secret_model, filter_model, svm, copy_model, dsl, logdir, target_class=2, epochs=100):
                
           
    targets_placeholder = tf.placeholder(tf.int32, shape=(None,), name='targets')

    height, width, channels = dsl.get_sample_shape()
    
    with tf.variable_scope("copy_model",reuse=tf.AUTO_REUSE):
        with tf.variable_scope("cw",reuse=tf.AUTO_REUSE):
            train_op, xadv, noise = cw(copy_model, y=targets_placeholder, eps=1.0, xshape=(cfg.batch_size, height, width, channels))    
            zero_noise = tf.assign(noise, tf.zeros_like(noise))
        
    var_list    = [v for v in tf.compat.v1.global_variables() if not 'cw' in v.name ]
    restorer    = tf.compat.v1.train.Saver(var_list=var_list)
    
    num_classes        = dsl.get_num_classes()
    invalid_class_idx  = dsl.get_num_classes()

    print "num classes: {}" .format(num_classes)
    print "invalid_class_idx: {}" .format(invalid_class_idx)
            
    with tf.compat.v1.Session(config=config) as sess:        
        sess.run(tf.compat.v1.global_variables_initializer())
        restorer.restore(sess, tf.train.latest_checkpoint(logdir))

        num_samples   = dsl.get_num_samples()
        valid_classes = dsl.get_num_classes()
        
        fsuccess    = 0
        psuccess    = 0
        asuccess    = 0
        
        true_acc     = compute_evaluation_measure(secret_model, sess, dsl, secret_model.sum_correct_prediction)        
        
        copy_acc     = compute_evaluation_measure(copy_model, sess, dsl, copy_model.sum_correct_prediction, add_class=(not cfg.ignore_invalid_class))
        
        print "Secret model Test Acc: {}" .format(true_acc)        
        print "Copy model Test Acc: {}" .format(copy_acc)            
                        
        for b in range(dsl.get_num_batches()):
            X, Y = dsl.load_next_batch(b)
                        
            clean_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, X)
            
            clean_pred_copy, clean_prob_copy = sess.run([copy_model.predictions, copy_model.prob] , feed_dict = { copy_model.X: X })

            L       = np.argsort(-clean_prob_copy, axis=-1)
            targets = L[ :, target_class-1 ]                                    
                 
            _ = sess.run(zero_noise)

            for _ in range(epochs):
                _, xadv_val = sess.run([train_op, xadv], {copy_model.X: X, targets_placeholder:targets})
                                                                     
            xadv_pred_copy = sess.run(copy_model.predictions , feed_dict = { copy_model.X: xadv_val })
            
            xadv_pred_secret = compute_svm_defended_model_preds_batch(sess, secret_model, filter_model, svm, xadv_val)
                                                            
            fsuccess += np.sum(np.logical_and(xadv_pred_secret==targets, xadv_pred_secret!=invalid_class_idx))       
            psuccess += np.sum(np.logical_and(xadv_pred_secret!=clean_pred_secret, xadv_pred_secret!=invalid_class_idx))   
            asuccess += np.sum(np.logical_and(targets == xadv_pred_copy, xadv_pred_copy!=invalid_class_idx))
        
        
        print 'CW Targeted-{} Partial Transferability success: {}' .format(target_class, psuccess/float(num_samples))
        print 'CW Targeted-{} Full Transferability success: {}' .format(target_class, fsuccess/float(num_samples))
        print 'Copy model CW Targeted-{} attack success:{}' .format(target_class, asuccess/float(num_samples))
        
def compute_attacks(secret_model, copy_model, test_dsl, logdir):
    
    eps = [ 0.1, 0.2, 0.25, 0.3 ]
    
    for eps_in in eps:
    
        print '\nUntargeted FGSM attack-{}' .format(eps_in)

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=1)

        generate_adversarial(xadv, secret_model, copy_model, test_dsl, logdir)     
        
    for eps_in in eps:        
        print '\nTargeted-{} FGSM attack-{}' .format(2, eps_in)                             #Second most likely label

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=-1)

        generate_adversarial_targeted(xadv, secret_model, copy_model, test_dsl, logdir)     
                
    num_classes = test_dsl.get_num_classes()
        
    for eps_in in eps:        
        print '\nTargeted-{} FGSM attack-{}' .format(num_classes, eps_in)                    #Least likely label

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=-1)

        generate_adversarial_targeted(xadv, secret_model, copy_model, test_dsl, logdir, target_class=num_classes)   
        

def compute_attacks_svm_filter(svm, secret_model, filter_model, copy_model, test_dsl, logdir):
        
    eps = [ 0.1, 0.2, 0.25, 0.3 ]
    
    for eps_in in eps:
    
        print '\nUntargeted FGSM attack-{}' .format(eps_in)

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=1)

        generate_adversarial_svm_filtered(xadv, secret_model, filter_model, svm, copy_model, test_dsl, logdir)               
            
    for eps_in in eps:
    
        print '\nTargeted-{} FGSM attack-{}' .format(2, eps_in)               #2nd most likely label

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=-1)

        generate_adversarial_svm_filtered_targeted(xadv, secret_model, filter_model, svm, copy_model, test_dsl, logdir)                                       
    num_classes = test_dsl.get_num_classes()
        
    for eps_in in eps:                
        print '\nTargeted-{} FGSM attack-{}' .format(num_classes, eps_in)     #Least likely label

        with tf.variable_scope("copy_model", reuse=True):
            xadv = fgm(copy_model, y=copy_model.labels, eps=eps_in, perturbation_multiplier=-1)

        generate_adversarial_svm_filtered_targeted(xadv, secret_model, filter_model, svm, copy_model, test_dsl, logdir,target_class=num_classes)   

        
# new train iter        
def train_copynet_iter(true_model, copy_model, train_dsl, valid_dsl, test_dsl, logdir_true, logdir_copy, ignore_vars=[], ignore_invalid_class=True):
    """ Trains the copy_model iteratively"""
    
    print "copynet will ignore invalid class"
    
    if cfg.defender_type is not None:
        oracle_contains_invalid_class = True
        if ignore_invalid_class:
            invalid_class_idx    = true_model.get_num_classes() - 1
            append_invalid_class = False
        else:
            invalid_class_idx    = -1
            append_invalid_class = True
    else:
        oracle_contains_invalid_class = False
        invalid_class_idx             = -1
        append_invalid_class          = False
        
    budget            = cfg.initial_seed+cfg.val_size+cfg.num_iter*cfg.k
    
    print "Total query budget: " , budget
    
    num_batches_tr   = train_dsl.get_num_batches()
    num_batches_test = test_dsl.get_num_batches()
    num_samples_test = test_dsl.get_num_samples()
    
    print "Noise dataset total train samples ", train_dsl.get_num_samples()
    print "Noise dataset total val samples ", valid_dsl.get_num_samples()

    
    num_classes      = true_model.get_num_classes()
    
    batch_size       = train_dsl.get_batch_size()
    
    noise_train_dsl                                  = DSLMarker(train_dsl)    
    noise_train_dsl_marked, noise_train_dsl_unmarked = noise_train_dsl.get_split_dsls()
    
    noise_val_dsl            = DSLMarker(valid_dsl)
    
    noise_val_dsl_marked, noise_val_dsl_unmarked = noise_val_dsl.get_split_dsls()
    
    orig_var_list = [v for v in tf.compat.v1.global_variables() if not v.name.startswith(copy_model.var_prefix)]    
    
    if len(ignore_vars) > 0:
        orig_var_list = [v for v in orig_var_list if not any([s in v.name for s in ignore_vars]) ]
            
    orig_saver    = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)    
    saver         = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints)

    train_writer = tf.summary.FileWriter(logdir_copy)
    train_writer.add_graph(true_model.get_graph())
    train_writer.add_graph(copy_model.get_graph())
     
    train_time = time.time()
        
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
        tf.logging.info('Model restored!')
                    
        val_label_counts = dict(list(enumerate([0] * num_classes)))
        
        # Mark validation set
        assert cfg.val_size % batch_size == 0
        
        for i in range(cfg.val_size / batch_size):
            trX, _       = valid_dsl.load_next_batch()
            trY          = get_predictions(sess, true_model, trX, one_hot=cfg.copy_one_hot)
            
            for k in range(len(trY)):
                if oracle_contains_invalid_class:
                    if ignore_invalid_class and np.argmax(trY[k]) != invalid_class_idx:                       
                        noise_val_dsl.mark(k + i*batch_size)
                        noise_val_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k][:-1] })

                    else:
                        noise_val_dsl.mark(k + i*batch_size)
                        noise_val_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })    
                else:
                    noise_val_dsl.mark(k + i*batch_size)
                    noise_val_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })                     

        
            for class_ in list(np.argmax(trY, -1)):
                val_label_counts[class_] += 1
        
        print "val label class dist: ", val_label_counts
        
        assert len(noise_val_dsl.marked) > 0, 'cannot validate! no validation samples. giving up.'
        
        pred_match = []
        
        
        assert cfg.initial_seed % batch_size == 0
        
        for i in range(cfg.initial_seed / batch_size):        
            trX, _        = train_dsl.load_next_batch()
            trY           = get_predictions(sess, true_model, trX, one_hot=cfg.copy_one_hot)                

            for k in range(len(trY)):                    
                if oracle_contains_invalid_class:
                    if ignore_invalid_class and np.argmax(trY[k]) != invalid_class_idx:                       
                        noise_train_dsl.mark(k + i*batch_size)
                        noise_train_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k][:-1] })

                    else:
                        noise_train_dsl.mark(k + i*batch_size)
                        noise_train_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })    
                else:
                    noise_train_dsl.mark(k + i*batch_size)
                    noise_train_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })
                    
                    
        
        Y_t     = get_labels(true_model, sess, test_dsl)
        
        print "Number of test samples" , len(Y_t)
        
        for it in range(cfg.num_iter+1):
            print "Processing iteration " , it+1                           
            label_counts = dict(list(enumerate([0] * num_classes)))
            
            sess.close()
            sess = tf.compat.v1.Session(config=config)            
            sess.run(tf.compat.v1.global_variables_initializer())
            
            orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
            
            saver         = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints)

            files = [file_ for file_ in os.listdir(logdir_copy) if os.path.isfile(os.path.join(logdir_copy, file_))]

            for file_ in files:
                os.remove(os.path.join(logdir_copy, file_))

            train_writer = tf.summary.FileWriter(logdir_copy)
            train_writer.add_graph(true_model.get_graph())
            train_writer.add_graph(copy_model.get_graph())

            gc.collect()
            
            print 'true model acc', compute_evaluation_measure(true_model, sess, test_dsl, true_model.sum_correct_prediction)
            print 'copy model acc', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
            
            print 'true model F1', compute_f1_measure(true_model, sess, test_dsl)
            print 'copy model F1', compute_f1_measure(copy_model, sess, test_dsl)
            
            exit      = False
            curr_loss = None
            best_f1  = None
            no_improvement = 0
            
            for epoch in range(cfg.copy_num_epochs):
                t_loss     = []
                epoch_time = time.time()
                
                print "\nProcessing epoch {} of iteration {}" .format(epoch+1, it+1)
                
                noise_train_dsl_marked.reset_batch_counter()
#                 noise_train_dsl_unmarked.reset_batch_counter()
                
                
                noise_train_dsl.shuffle_data()
               
                for i in range(noise_train_dsl_marked.get_num_batches()):
                    trX, _, trY_aux  = noise_train_dsl_marked.load_next_batch(return_idx=False, return_aux=True)                        
                    trY              = collect_aux_data(trY_aux, 'true_prob')
                    
                    trYhat, summary_str, loss, _, global_step = sess.run([
                                                      copy_model.prob,
                                                      copy_model.train_summary,
                                                      copy_model.mean_loss,
                                                      copy_model.train_op,
                                                      copy_model.global_step
                                                   ],
                                                   feed_dict={
                                                       copy_model.X: trX,
                                                       copy_model.labels: trY,
                                                       copy_model.dropout_keep_prob: cfg.dropout_keep_prob
                                                   })
                    t_loss += [loss]
                    
                    if epoch == 0:
                        for class_ in list(np.argmax(trY, -1)):
                            label_counts[class_] += 1

                    train_writer.add_summary(summary_str, global_step)
                    train_writer.flush()

                if (epoch+1) % cfg.copy_evaluate_every  == 0:
                    print('Epoch: {} Step: {} \tTrain Loss: {}'.format(epoch+1, global_step, np.mean(t_loss)))
                    print "Samples generated:", label_counts

                    curr_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
                    print "Test Accuracy (True Dataset): {}".format(curr_acc) 

                    curr_f1 = compute_f1_measure(copy_model, sess, test_dsl)
                    print "Test F1 (True Dataset): {}".format(curr_f1) 

                    val_acc = compute_evaluation_measure(copy_model, sess, noise_val_dsl_marked, copy_model.sum_correct_prediction, use_aux=True)
                    
                    val_f1 = compute_f1_measure(copy_model, sess, noise_val_dsl_marked, use_aux=True)
                    
                    if best_f1 is None or val_f1 > best_f1 :
                        best_f1 = val_f1
                        save_path = saver.save(sess, logdir_copy + '/model_step_%d' % (global_step))
                        print "Model saved in path: %s" % save_path
                        print "[BEST]",

                        no_improvement = 0
                    else:
                        no_improvement += 1
                        
                        if (no_improvement % cfg.copy_early_stop_tolerance) == 0:
                            if np.mean(t_loss) > 1.5:
                                no_improvement = 0
                            else:
                                exit = True

                    print "Valid Acc (Thief Dataset): {}".format(val_acc) 
                    print "Valid F1 (Thief Dataset): {}".format(val_f1) 
                    
                print "End of epoch {} (took {} minutes).".format(epoch+1, round((time.time() - epoch_time)/60, 2))
                
                if exit:
                    print "Number of epochs processed: {} in iteration {}" .format(epoch+1, it+1) 
                    break
                                
            saver.restore(sess, tf.train.latest_checkpoint(logdir_copy))

            # Log the best model for each iteration
            iter_save_path = os.path.join(logdir_copy, str(it))
            
            if not os.path.exists(iter_save_path):
                os.makedirs(iter_save_path)
            
            print 'Making directory:', iter_save_path 

            #for file_ in glob.glob(save_path + '*'):
                #shutil.copy(file_, iter_save_path)
                #print 'Copying file:', file_, 'To:', iter_save_path
            
            print 'copy model accuracy: ', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
            
            Y_copy  = get_labels(copy_model, sess, test_dsl)
            
            print "TA count" , np.sum(Y_t == Y_copy)
            print "Test agreement between source and copy model on true test dataset", np.sum(Y_t == Y_copy)/float(len(Y_t))
            
            if it+1 == cfg.num_iter+1:
                break
            
            X     = []
            Y     = []
            Y_idx = []
            idx   = []
            
            noise_train_dsl_unmarked.reset_batch_counter()
            
            
            print noise_train_dsl_unmarked.get_num_batches()
            
            for b in range(noise_train_dsl_unmarked.get_num_batches()):
                trX, _, tr_idx = noise_train_dsl_unmarked.load_next_batch(return_idx=True)
                
                trY, trY_idx = get_predictions(sess, copy_model, trX, labels=True)

                X.append(trX)
                Y.append(trY)
                Y_idx.append(trY_idx)
                idx.append(tr_idx)
            
            X      = np.concatenate(X)
            Y      = np.concatenate(Y)
            Y_idx  = np.concatenate(Y_idx)
            idx    = np.concatenate(idx)

            sss_time = time.time()
            
            # Core Set Construction
            if cfg.sampling_method == 'random':
                sss = RandomSelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'adversarial':
                sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model,K=len(Y))
            elif cfg.sampling_method == 'balancing':
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes)
            elif cfg.sampling_method == 'uncertainty':
                sss = UncertaintySelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'kmeans':
                sss = KMeansSelectionStrategy(cfg.k, Y, num_iter=cfg.kmeans_iter)
            elif cfg.sampling_method == 'kcenter':
                sss = KCenterGreedyApproach(cfg.k, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model))
            elif cfg.sampling_method == 'adversarial-balancing':
                sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model,K=len(Y))
                s   = sss.get_subset()
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)
            elif cfg.sampling_method == 'kcenter-balancing':
                sss = KCenterGreedyApproach(budget, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model))
                s   = sss.get_subset()
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)
	    elif cfg.sampling_method == 'uncertainty-balancing':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                s   = np.flip(s)
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)      
            elif cfg.sampling_method == 'uncertainty-adversarial':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model, perm=s)
            elif cfg.sampling_method == 'adversarial-kcenter':
                sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model, K=len(Y))
                s2 = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))
            elif cfg.sampling_method == 'uncertainty-adversarial-kcenter':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.phase1_fac*cfg.k, Y, X, sess, copy_model, K=(cfg.phase1_fac**2)*cfg.k, perm=s)
                s2 = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))       
            elif cfg.sampling_method == 'balancing-adversarial-kcenter':
                sss = BalancingSelectionStrategy(cfg.phase1_size, Y_idx, num_classes)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.phase2_size, Y, X, sess, copy_model, K=cfg.phase1_size, perm=s)
                s2  = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))       
            elif cfg.sampling_method == 'kcenter-adversarial':
                print "kcenter-adversarial method..."
                sss = KCenterAdvesarial(cfg.k, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model), X, sess, copy_model)
            else:
                raise Exception("sampling method {} not implemented" .format(cfg.sampling_method)) 
                
            
            s = sss.get_subset()
            
            if cfg.sampling_method in ['adversarial-kcenter', 'uncertainty-adversarial-kcenter']:
                s = s2[s]
            
            print "{} selection time:{} min" .format(cfg.sampling_method, round((time.time() - sss_time)/60, 2))

            if cfg.sampling_method != 'kmeans' and cfg.sampling_method != 'kcenter' :
                assert len(s) == cfg.k            
            
            s = np.unique(s)
            
            trX = [X[e] for e in s]
            true_trY, true_trY_idx = get_predictions(sess, true_model, trX, one_hot=cfg.copy_one_hot, labels=True)
            

            for i,k in enumerate(s):
                if oracle_contains_invalid_class:
                    if ignore_invalid_class and np.argmax(true_trY[i]) != invalid_class_idx:
                        noise_train_dsl.mark(idx[k], aux_data = { 'true_prob' : true_trY[i][:-1] })
                    else:
                        noise_train_dsl.mark(idx[k], aux_data = { 'true_prob' : true_trY[i] })
                else:
                    noise_train_dsl.mark(idx[k], aux_data = { 'true_prob' : true_trY[i] })           
            
            print "End of iteration ", it+1
        
        print "Copynet training completed in {} time" .format(round((time.time() - train_time)/3600, 2) )
        print "---Copynet trainning completed---"


def just_augment_dont_filter(sess, svm, filter_model, X, trY, invalid_class_idx, index):
    assert cfg.copy_one_hot == True
    assert cfg.svm_threshold is None, "Currently prob not supported"

    trY = np.argmax(trY, axis=-1).astype(int)
        
    if not index:
        trY = one_hot_labels(trY, invalid_class_idx+1)

    return trY


def filter_predictions_svm(sess, svm, filter_model, X, trY, invalid_class_idx, index):
    
    assert cfg.copy_one_hot == True
    
    assert cfg.svm_threshold is None, "Currently prob not supported"
    
    trZ          = get_metric_batch(filter_model, sess, X, filter_model.mean) #filter_model.z)
        
    if cfg.svm_threshold is not None:
        z_probs      = svm.predict_proba(trZ)
        neg_probs    = z_probs[:,0]
        mask         = neg_probs < cfg.svm_threshold
    else:
        mask         = (svm.predict(trZ)==1)
    
    if not index:
        assert trY.shape[1] == invalid_class_idx, 'Something wrong with invalid idx' 
        trY          = np.argmax(trY, axis=-1)
          
    filtered_trY = (mask*trY) + ((1-mask)* invalid_class_idx) 
    
    filtered_trY = filtered_trY.astype(int)
        
    if not index:
        filtered_trY = one_hot_labels(filtered_trY, invalid_class_idx+1)

    return filtered_trY





# train with Filter SVM
def train_copynet_iter_svm(secret_model, filter_model, svm, copy_model, train_dsl, valid_dsl, test_dsl, logdir_true, logdir_copy, ignore_vars=[], ignore_invalid_class=True):
    """ Trains the copy_model iteratively"""
    
    print "copynet will ignore invalid class"
    
    assert (cfg.defender_type is not None) 
    
    if ignore_invalid_class:
        invalid_class_idx    = secret_model.get_num_classes() 
        append_invalid_class = False
    else:
        invalid_class_idx    = -1
        append_invalid_class = True
        
    budget            = cfg.initial_seed+cfg.val_size+cfg.num_iter*cfg.k
    
    print "Total query budget: " , budget
    
    num_batches_tr   = train_dsl.get_num_batches()
    num_batches_test = test_dsl.get_num_batches()
    num_samples_test = test_dsl.get_num_samples()
    
    num_classes      = copy_model.get_num_classes() 
    
    label_classes    = secret_model.get_num_classes() + 1 
    
    batch_size       = train_dsl.get_batch_size()
    
    noise_train_dsl                                  = DSLMarker(train_dsl)    
    noise_train_dsl_marked, noise_train_dsl_unmarked = noise_train_dsl.get_split_dsls()
    
    noise_val_dsl            = DSLMarker(valid_dsl)   
    
    noise_val_dsl_marked, noise_val_dsl_unmarked = noise_val_dsl.get_split_dsls()
    
    orig_var_list = [v for v in tf.compat.v1.global_variables() if not v.name.startswith(copy_model.var_prefix)]    
    
    if len(ignore_vars) > 0:
        print "Ignoring variables" , ignore_vars
        orig_var_list = [v for v in orig_var_list if not any([s in v.name for s in ignore_vars]) ]
            
    orig_saver    = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints, var_list=orig_var_list)    
    saver         = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints)

    train_writer = tf.summary.FileWriter(logdir_copy)
    train_writer.add_graph(secret_model.get_graph())
    train_writer.add_graph(copy_model.get_graph())
     
    train_time = time.time()
        
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
        tf.logging.info('Model restored!')
                    
        val_label_counts = dict(list(enumerate([0] * label_classes)))
        
        # Mark validation set
        assert cfg.val_size % batch_size == 0
        
        for i in range(cfg.val_size / batch_size):
            trX, _       = valid_dsl.load_next_batch()
            trY          = get_predictions(sess, secret_model, trX, one_hot=cfg.copy_one_hot)                       
            
            # removed check for valid samples -- Soham
            trY          = just_augment_dont_filter(sess, svm, filter_model, trX, trY, invalid_class_idx, index=False)

            for k in range(len(trY)):
                if ignore_invalid_class:
                    if np.argmax(trY[k]) != invalid_class_idx:                       
                        noise_val_dsl.mark(k + i*batch_size)
                        noise_val_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k][:-1] })
                else:
                    noise_val_dsl.mark(k + i*batch_size)
                    noise_val_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })    

            for class_ in list(np.argmax(trY, -1)):
                val_label_counts[class_] += 1
                
            valid_class_counts = sum(val_label_counts.values())
            
            val_label_counts[invalid_class_idx] += cfg.val_size - valid_class_counts
        
        print "val label class dist: ", val_label_counts
        print "Total val count: ", sum(val_label_counts.values())
        
        if len(noise_val_dsl.marked) < 1:
            print "Cannot proceed copynet training! All the validation samples rejected"
            raise Exception("cannot validate! no validation samples. giving up.")            
            
        
        pred_match = []
        
        
        assert cfg.initial_seed % batch_size == 0
        
        no_seed_labels = True
        
        for i in range(cfg.initial_seed / batch_size):        
            trX, _     = train_dsl.load_next_batch()
            trY        = get_predictions(sess, secret_model, trX, one_hot=cfg.copy_one_hot)            
            trY        = filter_predictions_svm(sess, svm, filter_model, trX, trY, invalid_class_idx, index=False)

            for k in range(len(trY)):                
                if ignore_invalid_class:
                    if np.argmax(trY[k]) != invalid_class_idx:
                        no_seed_labels = False
                        noise_train_dsl.mark(k + i*batch_size)
                        noise_train_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k][:-1] })

                else:
                    noise_train_dsl.mark(k + i*batch_size)
                    noise_train_dsl.update(k + i*batch_size, aux_data={ 'true_prob' : trY[k] })    
        
        if no_seed_labels:
            print "Cannot proceed copynet training! All the seed samples rejected"
            raise Exception("Copynet Training failed due to seed samples rejected")
        
        Y_t     = get_labels(secret_model, sess, test_dsl)
        
        secret_preds_filtered = compute_svm_defended_model_preds(sess, secret_model, filter_model, svm, test_dsl)
        
        print "Number of test samples" , len(Y_t)
        
        for it in range(cfg.num_iter+1):
            print "Processing iteration " , it+1
                                       
            label_counts = dict(list(enumerate([0] * label_classes)))
            
            sess.close()
            sess = tf.compat.v1.Session(config=config)            
            sess.run(tf.compat.v1.global_variables_initializer())
            
            orig_saver.restore(sess, tf.train.latest_checkpoint(logdir_true))
            
            saver = tf.compat.v1.train.Saver(max_to_keep=cfg.num_checkpoints)

            files = [file_ for file_ in os.listdir(logdir_copy) if os.path.isfile(os.path.join(logdir_copy, file_))]

            for file_ in files:
                os.remove(os.path.join(logdir_copy, file_))

            train_writer = tf.summary.FileWriter(logdir_copy)
            train_writer.add_graph(secret_model.get_graph())
            train_writer.add_graph(copy_model.get_graph())

            gc.collect()
                        
            print 'true model acc', compute_evaluation_measure(secret_model, sess, test_dsl, secret_model.sum_correct_prediction)
            print 'copy model acc', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
            
            print 'true model F1', compute_f1_measure(secret_model, sess, test_dsl)
            print 'copy model F1', compute_f1_measure(copy_model, sess, test_dsl)
            
            exit      = False
            curr_loss = None
            best_f1  = None
            no_improvement = 0
            
            for epoch in range(cfg.copy_num_epochs):
                t_loss     = []
                epoch_time = time.time()
                
                print "\nProcessing epoch {} of iteration {}" .format(epoch+1, it+1)
                
                noise_train_dsl_marked.reset_batch_counter()
                noise_train_dsl.shuffle_data()
               
                for i in range(noise_train_dsl_marked.get_num_batches()):
                    trX, _, trY_aux  = noise_train_dsl_marked.load_next_batch(return_idx=False, return_aux=True)                        
                    trY              = collect_aux_data(trY_aux, 'true_prob')
                    
                    trYhat, summary_str, loss, _, global_step = sess.run([
                                                      copy_model.prob,
                                                      copy_model.train_summary,
                                                      copy_model.mean_loss,
                                                      copy_model.train_op,
                                                      copy_model.global_step
                                                   ],
                                                   feed_dict={
                                                       copy_model.X: trX,
                                                       copy_model.labels: trY,
                                                       copy_model.dropout_keep_prob: cfg.dropout_keep_prob
                                                   })
                    t_loss += [loss]
                    
                    if epoch == 0:
                        for class_ in list(np.argmax(trY, -1)):
                            label_counts[class_] += 1
                            
                        valid_class_counts = sum(label_counts.values())
                        
                        if cfg.k > 0:
                            label_counts[invalid_class_idx] += cfg.k - valid_class_counts
                        else:
                            label_counts[invalid_class_idx] += cfg.initial_seed - valid_class_counts
                            

                    train_writer.add_summary(summary_str, global_step)
                    train_writer.flush()

                if (epoch+1) % cfg.copy_evaluate_every  == 0:
                    print('Epoch: {} Step: {} \tTrain Loss: {}'.format(epoch+1, global_step, np.mean(t_loss)))
                    print "Samples generated upto {} iteration \n{} :" .format(it+1, label_counts)
                    print "Total samples generated so far ", sum(label_counts.values())
                    
                    
                    curr_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
                    print "Test Accuracy (True Dataset): {}".format(curr_acc) 

                    curr_f1 = compute_f1_measure(copy_model, sess, test_dsl)
                    print "Test F1 (True Dataset): {}".format(curr_f1) 

                    val_acc = compute_evaluation_measure(copy_model, sess, noise_val_dsl_marked, copy_model.sum_correct_prediction, use_aux=True)
                    
                    val_f1 = compute_f1_measure(copy_model, sess, noise_val_dsl_marked, use_aux=True)
                    
                    if best_f1 is None or val_f1 > best_f1 :
                        best_f1 = val_f1
                        save_path = saver.save(sess, logdir_copy + '/model_step_%d' % (global_step))
                        print "Model saved in path: %s" % save_path
                        print "[BEST]",

                        no_improvement = 0
                    else:
                        no_improvement += 1
                        
                        if (no_improvement % cfg.copy_early_stop_tolerance) == 0:
                            if np.mean(t_loss) > 1.5:
                                no_improvement = 0
                            else:
                                exit = True

                    print "Valid Acc (Thief Dataset): {}".format(val_acc) 
                    print "Valid F1 (Thief Dataset): {}".format(val_f1) 
                    
                print "End of epoch {} (took {} minutes).".format(epoch+1, round((time.time() - epoch_time)/60, 2))
                
                if exit:
                    print "Number of epochs processed: {} in iteration {}" .format(epoch+1, it+1) 
                    break
                                
            saver.restore(sess, tf.train.latest_checkpoint(logdir_copy))

            # Log the best model for each iteration
            iter_save_path = os.path.join(logdir_copy, str(it))
            if not os.path.exists(iter_save_path):
                os.makedirs(iter_save_path)
            print 'Making directory:', iter_save_path 

            #for file_ in glob.glob(save_path + '*'):
                #shutil.copy(file_, iter_save_path)
                #print 'Copying file:', file_, 'To:', iter_save_path
            
            print 'copy model accuracy: ', compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction, add_class=append_invalid_class)
            
            Y_copy  = get_labels(copy_model, sess, test_dsl)
            
            print "TA count" , np.sum(Y_t == Y_copy)
            print "Test agreement between source and copy model on true test dataset", np.sum(Y_t == Y_copy)/float(len(Y_t))
            
            print "Test agreement between defended secret and copy model on pure conf test set", np.sum(secret_preds_filtered == Y_copy)/float(len(secret_preds_filtered))
            
            if it+1 == cfg.num_iter+1:
                break
            
            X     = []
            Y     = []
            Y_idx = []
            idx   = []
            
            noise_train_dsl_unmarked.reset_batch_counter()
            
            #print noise_train_dsl_unmarked.get_num_batches()
            
            for b in range(noise_train_dsl_unmarked.get_num_batches()):
                trX, _, tr_idx = noise_train_dsl_unmarked.load_next_batch(return_idx=True)
                trY, trY_idx = get_predictions(sess, copy_model, trX, labels=True)

                X.append(trX)
                Y.append(trY)
                Y_idx.append(trY_idx)
                idx.append(tr_idx)
            
            X      = np.concatenate(X)
            Y      = np.concatenate(Y)
            Y_idx  = np.concatenate(Y_idx)
            idx    = np.concatenate(idx)
            
            sss_time = time.time()
            # Core Set Construction
            if cfg.sampling_method == 'random':
                sss = RandomSelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'adversarial':
                sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model,K=len(Y))
            elif cfg.sampling_method == 'balancing':
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes)
            elif cfg.sampling_method == 'uncertainty':
                sss = UncertaintySelectionStrategy(cfg.k, Y)
            elif cfg.sampling_method == 'kmeans':
                sss = KMeansSelectionStrategy(cfg.k, Y, num_iter=cfg.kmeans_iter)
            elif cfg.sampling_method == 'kcenter':
                sss = KCenterGreedyApproach(cfg.k, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model))
                #sss = KCenterGreedyApproach(cfg.k, Y, true_initial_centers(sess, noise_train_dsl_marked))
            elif cfg.sampling_method == 'adversarial-balancing':
                sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model,K=len(Y))
                s   = sss.get_subset()
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)
            elif cfg.sampling_method == 'kcenter-balancing':
                sss = KCenterGreedyApproach(budget, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model))
                s   = sss.get_subset()
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)
            elif cfg.sampling_method == 'uncertainty-balancing':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                s   = np.flip(s)
                sss = BalancingSelectionStrategy(cfg.k, Y_idx, num_classes, s)      
            elif cfg.sampling_method == 'uncertainty-adversarial':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.k, Y, X, sess, copy_model, perm=s)
            elif cfg.sampling_method == 'adversarial-kcenter':
                sss = AdversarialSelectionStrategy(budget, Y, X, sess, copy_model, K=len(Y))
                s2 = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))
            elif cfg.sampling_method == 'uncertainty-adversarial-kcenter':
                sss = UncertaintySelectionStrategy(len(Y), Y)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.phase1_fac*cfg.k, Y, X, sess, copy_model, K=(cfg.phase1_fac**2)*cfg.k, perm=s)
                s2 = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))       
            elif cfg.sampling_method == 'balancing-adversarial-kcenter':
                sss = BalancingSelectionStrategy(cfg.phase1_size, Y_idx, num_classes)
                s   = sss.get_subset()
                sss = AdversarialSelectionStrategy(cfg.phase2_size, Y, X, sess, copy_model, K=cfg.phase1_size, perm=s)
                s2  = np.array(sss.get_subset())
                sss = KCenterGreedyApproach(cfg.k, Y[s2], get_initial_centers(sess, noise_train_dsl_marked, copy_model))       
            elif cfg.sampling_method == 'kcenter-adversarial':
                print "kcenter-adversarial method..."
                sss = KCenterAdvesarial(cfg.k, Y, get_initial_centers(sess, noise_train_dsl_marked, copy_model), X, sess, copy_model)
            else:
                raise Exception("sampling method {} not implemented" .format(cfg.sampling_method)) 
                
            
            s        = sss.get_subset()
            
            if cfg.sampling_method in ['adversarial-kcenter', 'uncertainty-adversarial-kcenter']:
                s = s2[s]
            
            print "{} selection time:{} min" .format(cfg.sampling_method, round((time.time() - sss_time)/60, 2))

            if cfg.sampling_method != 'kmeans' and cfg.sampling_method != 'kcenter' :
                assert len(s) == cfg.k
            
            s = np.unique(s)
            
            trX = [X[e] for e in s]
            
            true_trY, true_trY_idx = get_predictions(sess, secret_model, trX, one_hot=cfg.copy_one_hot, labels=True)        
            true_trY               = filter_predictions_svm(sess, svm, filter_model, trX, true_trY, invalid_class_idx, index=False)
            true_trY_idx           = np.argmax(true_trY, axis=-1)

            for i,k in enumerate(s):
                if ignore_invalid_class:
                    if np.argmax(true_trY[i]) != invalid_class_idx:
                        noise_train_dsl.mark(idx[k], aux_data = { 'true_prob' : true_trY[i][:-1] })
                else:
                    noise_train_dsl.mark(idx[k], aux_data = { 'true_prob' : true_trY[i] })
                        
            print "End of iteration ", it+1

            true_trY_idx_unique = np.unique(list(true_trY_idx))

            if len(true_trY_idx_unique) == 1 and true_trY_idx_unique[0] == invalid_class_idx:
                print "Cannot proceed copynet training! All the selected samples rejected. Breaking..."
                break
        
        print "Copynet training completed in {} time" .format(round((time.time() - train_time)/3600, 2) )
        print "---Copynet training completed---"
                
