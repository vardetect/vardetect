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

from __future__ import absolute_import
from __future__ import division
import sys, time, os, logging, time
import numpy as np
import shutil
from utils.model import *
from utils.class_loader import *
from utils.helper import *
from dsl.uniform_dsl import UniformDSL
from dsl.imagenet_dsl import ImagenetDSL
from models.defended_model import *
import tensorflow as tf
from cfg import cfg
from sklearn.svm import SVC

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

for key, value in tf.flags.FLAGS.__flags.items():
    try:
        logging.info("{} {}" .format(key, value.value) )
        print "{} {}" .format(key, value.value) 
    except AttributeError:
        logging.info("{} {}" .format(key, value) )
        print "{} {}" .format(key, value)

tf.set_random_seed(cfg.seed)        
        
cfg.source_model = 'deepcnn'
cfg.copy_model = 'deepcnn'
assert cfg.true_dataset is not None
cfg.noise_dataset = 'imagenet'
cfg.copy_one_hot = True

logdir_true = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, 'true')
    
if cfg.defender_type is not None:    
    suffix          = '_'.join( [ cfg.defender_type.split('_')[0] , str(cfg.latent_dim) ] )
             
    logdir_defender = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, suffix )
            
    print logdir_defender    
    
noise_dataset = cfg.noise_dataset

if cfg.num_to_keep is not None:
    noise_dataset = noise_dataset + '-' + str(cfg.num_to_keep)

if cfg.extract_model_activethief or cfg.transfer_attack_activethief: 
    assert cfg.num_iter is not None
    
    cfg.val_size     = int(0.2*cfg.query_budget)
    
    if cfg.num_iter == 0:
        cfg.k  = 0
        cfg.initial_seed = int(0.8*cfg.query_budget)
    else:
        cfg.k  = int( (0.7*args.query_budget)/cfg.num_iter)
        cfg.initial_seed = int(0.1*cfg.query_budget)
    
    train_size = int(0.8*float(cfg.num_to_keep))
    val_size   = int(0.2*float(cfg.num_to_keep))
    
    noise_dataset = "{}-{}-{}+{}+{}" .format( noise_dataset, cfg.sampling_method, cfg.initial_seed, cfg.val_size , cfg.num_iter * cfg.k )
    
if cfg.copy_one_hot:
    api_retval = 'onehot'
else:
    api_retval = 'softmax'


if cfg.defender_type is not None:        
    logdir_copy = os.path.join('modeldir' , 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, noise_dataset, cfg.defender_type )

    logdir_papernot_copy = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, 'jbda', '{}-{}-{}' .format( cfg.jtype, cfg.eps, cfg.query_budget ), cfg.defender_type)                    

    logdir_tramer_copy = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, 'tramer', '{}-{}' .format( cfg.sampling_method, cfg.query_budget ) , cfg.defender_type)                        
else:    
    logdir_copy = os.path.join('modeldir' , 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, noise_dataset, 'undefended' )

    logdir_papernot_copy = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, 'jbda', '{}-{}-{}' .format( cfg.jtype, cfg.eps, cfg.query_budget ) , 'undefended')                        

    logdir_tramer_copy = os.path.join('modeldir', 'images', cfg.source_model, cfg.true_dataset, api_retval, cfg.copy_model, 'tramer', '{}-{}' .format( cfg.sampling_method, cfg.query_budget ) , 'undefended')                        
    
if cfg.defender_type is not None:        
    if cfg.use_recon_input:
        logdir_copy_defended          = os.path.join(logdir_copy , 'recon')                                 
        logdir_papernot_copy_defended = os.path.join(logdir_papernot_copy , 'recon')
        logdir_tramer_copy_defended = os.path.join(logdir_tramer_copy , 'recon')
    else:
        logdir_copy_defended          = os.path.join(logdir_copy , 'clean')
        logdir_papernot_copy_defended = os.path.join(logdir_papernot_copy , 'clean')
        logdir_tramer_copy_defended   = os.path.join(logdir_tramer_copy , 'clean')

true_dataset_dsl = load_dataset(cfg.true_dataset) 

if not cfg.train_source_model:
    shuffle_train_each_epoch = False

def create_loaders(ds, batch_size, sample_limit=None, resize=None, normalize_channels=None):
    train_dsl = ds(batch_size = batch_size, mode='train', shuffle_each_epoch=True, seed=cfg.seed, sample_limit=sample_limit, resize=resize, normalize_channels=normalize_channels )
    val_dsl   = ds(batch_size = batch_size, mode='val', shuffle_each_epoch=False, seed=cfg.seed, sample_limit=sample_limit, resize=resize, normalize_channels=normalize_channels )
    test_dsl  = ds(batch_size = batch_size, mode='test', shuffle_each_epoch=False, sample_limit=sample_limit, resize=resize, normalize_channels=normalize_channels )
    return train_dsl, val_dsl, test_dsl

train_dsl, val_dsl, test_dsl = create_loaders(true_dataset_dsl, cfg.batch_size)

sample_shape = train_dsl.get_sample_shape()
width, height, channels = sample_shape
is_multilabel = train_dsl.is_multilabel()
resize = (width, height)
num_classes = train_dsl.get_num_classes()

if cfg.extract_model_activethief:

    noise_dataset_dsl = load_dataset(cfg.noise_dataset) 

    count = 1

    while True:
        try:
            print "Loading noise data. Attempt {}" .format(count)
            if noise_dataset_dsl == UniformDSL:
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', shape=sample_shape, sample_limit=val_size, seed=cfg.seed)
            elif noise_dataset_dsl == ImagenetDSL and cfg.num_to_keep is not None:
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', num_to_keep=val_size, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
            else:
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', seed=cfg.seed)
            break
        except MemoryError as e:
            if count==5:
                raise Exception("Memory error could not be resolved using time delay")
            else:
                print "Loading data failed. Waiting for 5 min.."
                time.sleep(300)        
            count = count + 1

    print "Data loaded"

    _, _, noise_channels = noise_val_dsl.get_sample_shape()
    normalize_channels = (channels == 1 and noise_channels != 1)
    
    count = 1

    while True:
        try:
            print "Loading data. Attempt {}" .format(count)
            if noise_dataset_dsl == UniformDSL:
                noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', shape=sample_shape, sample_limit= train_size, seed=cfg.seed)
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', shape=sample_shape, sample_limit=val_size, seed=cfg.seed)
            elif noise_dataset_dsl == ImagenetDSL and cfg.num_to_keep is not None:
                noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', resize=resize, normalize_channels=normalize_channels, num_to_keep=train_size, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', resize=resize, normalize_channels=normalize_channels, num_to_keep=val_size, start_batch=cfg.subsampling_start_batch, end_batch=cfg.subsampling_end_batch, seed=cfg.seed)
            else:
                noise_train_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='train', resize=resize, normalize_channels=normalize_channels)
                noise_val_dsl = noise_dataset_dsl(batch_size = cfg.batch_size, mode='val', resize=resize, normalize_channels=normalize_channels, seed=cfg.seed)
            break
        except MemoryError as e:
            if count==5:
                raise Exception("Memory error could not be resolved using time delay")
            else:
                print "Loading data failed. Waiting for 5 min.."
                time.sleep(300)        
            count = count + 1

    print "Noise data loaded"        

source_model_type  = load_model(cfg.source_model)
copy_model_type    = load_model(cfg.copy_model)

if cfg.train_source_model:
    
    tf.reset_default_graph()
        
    with tf.variable_scope("true_model"):
        true_model = source_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel, var_prefix='true_model')

        true_model.print_trainable_parameters()

        true_model.print_arch()

    logging.info("Training source model...")
    t = time.time()
    shutil.rmtree(logdir_true, ignore_errors=True, onerror=None)
    train_model(model=true_model, train_dsl=train_dsl, val_dsl=val_dsl, logdir=logdir_true)
    logging.info("Training source model completed {} min" .format( round((time.time() - t)/60, 2)  ) )

if cfg.defender_type is not None:
    if "dae" in cfg.defender_type:
        dae_type  = load_dae_type(cfg.true_dataset)

    if cfg.defender_type == "dcgan":
        gan_type   = load_gan_type(cfg.true_dataset)

    if "vae" in cfg.defender_type:
        vae_type   = load_vae_type(cfg.true_dataset)

if cfg.train_defender:       
    shutil.rmtree(logdir_defender, ignore_errors=True, onerror=None)
    
    tf.reset_default_graph()
    
    with tf.variable_scope("true_model"):
        true_model = source_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel, is_training=False, var_prefix='true_model')
        
    
    if cfg.defender_type is not None:   
        
        if "vae" in cfg.defender_type:
            print('hi, gonna load VAE')
            with tf.variable_scope("vae"):
                vae = vae_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, z_size=cfg.latent_dim, noise_mean=cfg.noise_mean)

            print('hi, gonna train VAE')
            train_vae(model=vae, train_dsl=train_dsl, val_dsl=val_dsl, logdir=logdir_defender, logdir_true=logdir_true)

        else:
            raise Exception("Defender not recoginized")
        
ignore_vars = ['reconstructor', 'dcgan_1']
                                       
# Set appropriate model dir
if cfg.defender_type is not None:    
    logdir_oracle       = logdir_defender 
    logdir_sub          = logdir_copy_defended
    logdir_sub_papernot = logdir_papernot_copy_defended
else:
    print "All dir set to work with undefended"
    logdir_oracle       = logdir_true
    logdir_sub          = logdir_copy
    logdir_sub_papernot = logdir_papernot_copy

print "Oracle model be loaded from: ", logdir_oracle

        
def create_graph(is_train=False):    
    tf.reset_default_graph()
        
    with tf.variable_scope("true_model", reuse=tf.AUTO_REUSE):
        true_model = source_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=num_classes, multilabel=is_multilabel, is_training=False, var_prefix='true_model')
    
        true_model.print_arch()
        print "\n"
    
    print "True Model Test Accuracy: ", evaluate(model=true_model, dsl=test_dsl, logdir=logdir_true)        
        
    if cfg.defender_type is not None:
        
        if "dae" in cfg.defender_type:
            
            with tf.variable_scope("dae", reuse=tf.AUTO_REUSE):
                filter_model = dae_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, z_size=cfg.latent_dim, is_training=False)
                                
            print "defender path", logdir_defender
            rlosses_true  = compute_reconstruction_losses(filter_model, val_dsl, logdir_defender)     
             
            print_dist_params(rlosses_true)
            
            threshold  = np.percentile( rlosses_true, 99 )        
            
            print "threshold for the recon loss is set to {} for {} percentile" .format( threshold, 99 )         
                   
        elif "vae" in cfg.defender_type:
            
            with tf.variable_scope("vae", reuse=tf.AUTO_REUSE):
                filter_model = vae_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, is_training=False, z_size=cfg.latent_dim)
                
            rlosses_true  = compute_reconstruction_losses(filter_model, val_dsl, logdir_defender)       
                                       
            print_dist_params(rlosses_true)
            
            threshold  = np.percentile( rlosses_true, 99 )       
            
            print "threshold for the recon loss is set to {} for {} percentile" .format( threshold, 99 )        
                        
        elif "dcgan" in cfg.defender_type :
            with tf.variable_scope("dcgan", reuse=tf.AUTO_REUSE):
                filter_model = gan_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, z_size=cfg.latent_dim, is_training=False)
            
            print "Recon epochs for GAN are set to {}" .format( filter_model.recon_epochs )
            
            threshold = [ cfg.disc_threshold, cfg.recon_threshold ]
            
            print "threshold of the disc for fake sample is set to {}" .format( cfg.disc_threshold )            
            print "threshold for the recon loss is set to {}" .format( cfg.recon_threshold )                                         
        else:
            raise Exception("Implement for other defense type")
        
        if cfg.use_recon_input:
            print "Creating defender that uses recon sample for oracle prediction"
            with tf.variable_scope("true_model", reuse=tf.AUTO_REUSE):    
                defended_model = DefenseModel(true_model, filter_model, threshold=threshold, defender_type=cfg.defender_type, use_recon_input=cfg.use_recon_input) 
        else:        
            print "Creating defender that uses clean sample for oracle prediction"
            with tf.variable_scope( cfg.defender_type.split("_")[0] , reuse=tf.AUTO_REUSE):    
                defended_model = DefenseModel(true_model, filter_model, threshold=threshold, defender_type=cfg.defender_type, use_recon_input=cfg.use_recon_input)   
                                
        start_time = time.time()
        print "Defended Model Test Accuracy (Recon Filter): ", evaluate(model=defended_model, dsl=test_dsl, logdir=logdir_defender, ignore_vars=ignore_vars)
        
        print "Total time {} min for evaluation" .format(round((time.time() - start_time)/60, 2) )
        
        oracle_model = defended_model
    else:
        oracle_model = true_model
        
    copy_num_classes    = num_classes
    
    if cfg.defender_type is not None:
        if not cfg.ignore_invalid_class:
            copy_num_classes    = num_classes + 1
                
    with tf.variable_scope("copy_model"):
        copy_model = copy_model_type(batch_size=cfg.batch_size, height=height, width=width, channels=channels, num_classes=copy_num_classes, multilabel=is_multilabel, is_training=is_train, var_prefix='copy_model', learning_rate =cfg.learning_rate)
                                       
        copy_model.print_arch()
    
    return oracle_model, copy_model        


if cfg.extract_model_activethief:
    
    oracle_model , copy_model = create_graph(True)
    
    print "Copymodel will be saved in: ", logdir_sub

         
    if cfg.defender_type is not None: #and 'svm' in cfg.defender_type:        
                                        
        svm = load_svm( oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, noise_train_dsl, logdir_oracle )    
        print "deleting the dir {}" .format( logdir_copy )        
        
        print "Training Copy Model with svm filter.."
        
        train_copynet_iter_svm( oracle_model.true_model, oracle_model.filter_model, svm, copy_model, noise_train_dsl, noise_val_dsl, test_dsl, logdir_oracle, logdir_sub, ignore_vars = ignore_vars, ignore_invalid_class=cfg.ignore_invalid_class)    
    else:
        
        print "Training Copy Model for undefended oracle.."
        
        train_copynet_iter(oracle_model, copy_model, noise_train_dsl, noise_val_dsl, test_dsl, logdir_oracle, logdir_sub, ignore_vars = ignore_vars, ignore_invalid_class=cfg.ignore_invalid_class)
                
                
if cfg.transfer_attack_activethief:
    
    print "---------Extraction Attack START---------"
    
    oracle_model , copy_model = create_graph(True)
    
    print "Copymodel will be loaded from: ", logdir_sub
    
    print "Copy Model accuracy extracting secret model"
    evaluate(model=copy_model, dsl=test_dsl, logdir=logdir_sub, add_class=( not cfg.ignore_invalid_class) )

    print "Extraction transfer attack on secret model"

    if (cfg.defender_type is not None) and 'svm' in cfg.defender_type:
        
        print "Loading svm.."
        
        if cfg.filtered_attack:
            svm = load_svm( oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, None, logdir_oracle )                 
            compute_attacks_svm_filter(svm, oracle_model.true_model, oracle_model.filter_model, copy_model, test_dsl, logdir_sub) 
        else:
            print "Running transferability attack of extracted model when defense was in place without filtering"
            compute_attacks(oracle_model.true_model, copy_model, test_dsl, logdir_sub)
    else:
        compute_attacks(oracle_model, copy_model, test_dsl, logdir_sub)
    
    print "---------Extraction Attack END---------"


if cfg.extract_model_jbda or cfg.transfer_attack_jbda:
    cfg.batch_size    = 150
    epochs            = 100

    # Setting initial seed sample
    if cfg.true_dataset=='gtsr':
        train_size  = 430
    else:
        train_size  = 150   
    
    print "Copymodel will be saved in: ", logdir_sub_papernot
    
    train_dsl, val_dsl, test_dsl = create_loaders(true_dataset_dsl, cfg.batch_size)
    
    print "Test batch_size: {} Test sample_shape: {}" .format( test_dsl.get_batch_size(), test_dsl.get_sample_shape() )
    
    if cfg.extract_model_jbda:

        oracle_model, copy_model = create_graph(is_train=True)
        
        X_test, Y_test = test_dsl.data, test_dsl.labels

        X_sub, Y_sub, mod_test_dsl = get_test_subset(test_dsl, train_size)
        
        print "Creating initial seed samples: ", X_sub.shape, Y_sub.shape 
                
        print "Mod test dsl excluding initial seed samples: ", mod_test_dsl.data.shape, mod_test_dsl.labels.shape 

        if (cfg.defender_type is not None) and ('svm' in cfg.defender_type):        

            print "Loading svm.."

            svm = load_svm( oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, None, logdir_oracle ) 

            print "\nTraining copynet using jbda-{} with svm filter" .format( cfg.jtype )

            train_jacobian_svm(oracle_model.true_model, oracle_model.filter_model, svm, copy_model, X_sub, Y_sub, mod_test_dsl, logdir_oracle, logdir_sub_papernot, epochs=epochs, query_budget=cfg.query_budget, ignore_vars=ignore_vars, jtype=cfg.jtype, eps=cfg.eps)            
        else:
            print "Training copynet using jbda-{}" .format( cfg.jtype )

            train_jacobian(oracle_model, copy_model, X_sub, Y_sub, mod_test_dsl, logdir_oracle, logdir_sub_papernot, epochs=epochs, query_budget=cfg.query_budget, ignore_vars=ignore_vars, jtype=cfg.jtype, eps=cfg.eps)        


    if cfg.transfer_attack_jbda:

        oracle_model , copy_model = create_graph()

        print "---------JbDA Attack START---------"

        print "Copy model will be loaded from: ", logdir_sub_papernot
        
        copy_acc = evaluate(model=copy_model, dsl=test_dsl, logdir=logdir_sub_papernot, ignore_vars=ignore_vars, add_class=(not cfg.ignore_invalid_class) )
        
        print "Copy Model accuracy extracting secret model: ", copy_acc
        
        print "Extraction transfer attack on secret model"

        if (cfg.defender_type is not None) and ('svm' in cfg.defender_type):        
            print "Loading svm.."
            
            if cfg.filtered_attack:
                svm = load_svm( oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, None, logdir_oracle )                 
                compute_attacks_svm_filter(svm, oracle_model.true_model, oracle_model.filter_model, copy_model, test_dsl, logdir_sub_papernot)
            else:
                print "Running transferability attack of extracted model when defense was in place without filtering"
                compute_attacks(oracle_model.true_model, copy_model, test_dsl, logdir_sub_papernot)
        else:        
            compute_attacks(oracle_model, copy_model, test_dsl, logdir_sub_papernot)

        print "---------JbDA Attack END---------"    


        
if cfg.extract_model_tramer:
    oracle_model, copy_model = create_graph(is_train=True)
        
    print "Extracting model using {} attack of Tramer" .format(cfg.sampling_method)    
        
    print "Copy model will be loaded from: ", logdir_tramer_copy        
    
    print "Total query budget", cfg.query_budget
    
    if (cfg.defender_type is not None ) and 'svm' in cfg.defender_type:        
        pass
        print 'before load_svm'
        svm = load_svm(oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, None, logdir_oracle)    
        
        print "Training Copy Model using Tramer attack with svm filter {}" .format(cfg.sampling_method)
        print 'logdir_oracle', logdir_oracle
        print 'logdir_tramer_copy', logdir_tramer_copy

        train_copynet_tramer_svm(oracle_model.true_model, oracle_model.filter_model, svm, copy_model, test_dsl, logdir_oracle, logdir_tramer_copy)    
    else:        
        print "Training Copy Model using Tramer attack {}" .format( cfg.sampling_method )       
        train_copynet_tramer(oracle_model, copy_model, test_dsl, logdir_oracle, logdir_tramer_copy) 
        
        
        
if cfg.transfer_attack_tramer:

    oracle_model , copy_model = create_graph()

    print "---------Tramer Transferability attack START---------"

    print "Copy model will be loaded from: ", logdir_tramer_copy

    copy_acc = evaluate(model=copy_model, dsl=test_dsl, logdir=logdir_tramer_copy, ignore_vars=ignore_vars, add_class=(not cfg.ignore_invalid_class) )

    print "Copy Model accuracy extracting secret model: ", copy_acc

    print "Extraction transfer attack on secret model"

    if (cfg.defender_type is not None) and ('svm' in cfg.defender_type):        
        print "Loading svm.."

        if cfg.filtered_attack:
            svm = load_svm( oracle_model.filter_model, oracle_model.true_model, train_dsl, val_dsl, test_dsl, None, logdir_oracle )                 
            compute_attacks_svm_filter(svm, oracle_model.true_model, oracle_model.filter_model, copy_model, test_dsl, logdir_tramer_copy)
        else:
            print "Running transferability attack of extracted model when defense was in place without filtering"
            compute_attacks(oracle_model.true_model, copy_model, test_dsl, logdir_tramer_copy)
    else:        
        compute_attacks(oracle_model, copy_model, test_dsl, logdir_tramer_copy)

    print "---------Tramer Transferability attack END---------"    
