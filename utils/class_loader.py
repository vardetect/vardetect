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

def load_dataset(dataset):
    dsl = None
    
    if dataset == 'mnist':
        from dsl.mnist_dsl import MNISTDSL
        dsl = MNISTDSL

    elif dataset == 'gtsr':
        from dsl.gtsr_dsl import GTSRDSL
        dsl = GTSRDSL

    elif dataset == 'cifar':
        from dsl.cifar_dsl import CifarDSL
        dsl = CifarDSL

    elif dataset == 'celeba':
        from dsl.celeba_dsl import CelebADSL
        dsl = CelebADSL

    elif dataset == 'streetview':
        from dsl.streetview_dsl import StreetViewDSL
        dsl = StreetViewDSL

    elif dataset == 'cub':
        from dsl.cub_dsl import CUBDSL
        dsl = CUBDSL

    elif dataset == 'omniglot':
        from dsl.omniglot_dsl import OmniglotDSL
        dsl = OmniglotDSL

    elif dataset == 'catdog':
        from dsl.catdog_dsl import CatDogDSL
        dsl = CatDogDSL

    elif dataset == 'fashionmnist':
        from dsl.fashionmnist_dsl import FashionMNISTDSL
        dsl = FashionMNISTDSL

    elif dataset == 'imagenet':
        from dsl.imagenet_dsl import ImagenetDSL
        dsl = ImagenetDSL
    
    elif dataset == 'uniform':
        from dsl.uniform_dsl import UniformDSL
        dsl = UniformDSL
    
    elif dataset == 'imdb':
        from dsl.imdb_dsl import IMDBDSL
        dsl = IMDBDSL

    elif dataset == 'wiki':
        from dsl.wiki_dsl import WikiDSL
        dsl = WikiDSL        

    elif dataset == 'randomword':
        from dsl.randomword_dsl import RandomWordDSL
        dsl = RandomWordDSL                

    elif dataset == 'agnews':
        from dsl.agnews_dsl import AGNewsDSL
        dsl = AGNewsDSL        

    elif dataset == 'snli':
        from dsl.snli_dsl import SNLIDSL
        dsl = SNLIDSL                

    elif dataset == 'qc':
        from dsl.qc_dsl import QCDSL
        dsl = QCDSL

    elif dataset == 'sms':
        from dsl.sms_dsl import SMSSpamDSL
        dsl = SMSSpamDSL        
        
    elif dataset == 'mr':
        from dsl.mr_dsl import MRDSL
        dsl = MRDSL        
        
    else:
        raise Exception("Dataset {} could not be loaded" .format( dataset ) ) 
    
    return dsl


def load_dae_type(dataset):
    
    if dataset == 'mnist':
        from models.autoencoder import MnistDAE
        dae = MnistDAE        

    elif dataset == 'gtsr':
        from models.autoencoder import GTSRDAE
        dae = GTSRDAE        

    elif dataset == 'cifar':
        from models.autoencoder import CifarDAE
        dae = CifarDAE        
    
    elif dataset == 'fashionmnist':
        from models.autoencoder import MnistDAE
        dae = MnistDAE            
    
    return dae
    
def load_gan_type(dataset):
    
    if dataset == 'mnist':
        from models.dcgan import MnistDCGAN
        gan = MnistDCGAN        

    elif dataset == 'gtsr':
        from models.dcgan import GTSRDCGAN
        gan = GTSRDCGAN
        
    elif dataset == 'cifar':
        from models.dcgan import CifarDCGAN
        gan = CifarDCGAN
    
    elif dataset == 'fashionmnist':
        from models.dcgan import FashionMnistDCGAN
        gan = FashionMnistDCGAN        
    
    else:
        raise Exception("Gan not defined")
    
    return gan
    

def load_vae_type(dataset):
    
    if dataset == 'mnist':
        from models.vae import NewHSVNVAE28
        vae = NewHSVNVAE28        

    elif dataset == 'gtsr':
        from models.vae import NewHSVNVAE32
        vae = NewHSVNVAE32
        
    elif dataset == 'cifar':
        from models.vae import CifarVAE
        vae = CifarVAE
    
    elif dataset == 'fashionmnist':
        from models.vae import NewHSVNVAE28
        vae = NewHSVNVAE28       
    
    elif dataset == 'streetview':
        from models.vae import NewHSVNVAE32
        vae = NewHSVNVAE32       
    
    elif dataset == 'cub':
        from models.vae import NewHSVNVAE32
        vae = NewHSVNVAE32       
    
    elif dataset == 'omniglot':
        from models.vae import NewHSVNVAE32
        vae = NewHSVNVAE32       
    
    elif dataset == 'catdog':
        from models.vae import NewHSVNVAE32
        vae = NewHSVNVAE32       
    
    else:
        raise Exception("VAE not defined")
    
    return vae
    
    
    
def load_optimizer(optimizer):    
    
    if optimizer == 'adagrad':
        optimizer  = tf.train.AdagradOptimizer(cfg.learning_rate) # None
    elif optimizer == 'adam':
        optimizer  = tf.train.AdamOptimizer() # None
    else:
        assert optimizer is None
        optimizer = None    
    
    return optimizer

def load_model(model_class):
    
    model = None
    
    if model_class == 'deepcnn':
        from models.deepcnn import DeepCNN
        model = DeepCNN
    elif model_class == 'deepmlp':
        from models.deepmlp import DeepMLP
        model = DeepMLP
    elif model_class == 'shallowcnn':
        from models.shallowcnn import ShallowCNN
        model = ShallowCNN
    elif model_class == 'shallowmlp':
        from models.shallowmlp import ShallowMLP
        model = ShallowMLP
    elif model_class.startswith('cnn_'):
        from models.cnn import CNN
        blocks, convs_in_block = model_class.strip().split('_')[1:]
        blocks, convs_in_block = int(blocks), int(convs_in_block)
    
        class CNNWrapper(CNN):
            def __init__(self, *args, **kwargs):
                super(CNNWrapper, self).__init__(*args, convs_in_block=convs_in_block, num_filters=[32, 64, 128, 256][:blocks], **kwargs)

        return CNNWrapper
    elif model_class == 'textcnn':
        from models.textcnn import TextCNN
        model = TextCNN
    elif model_class == 'textrnn':
        from models.rnn import RNN
        model = RNN                
    elif model_class == 'vgg16':
        from models.vgg16 import VGG16
        model = VGG16
    elif model_class == 'vgg19':
        from models.vgg19 import VGG19
        model = VGG19

    else:
        raise Exception("Model {} could not be loaded" .format( model_class ) ) 
        
    return model
