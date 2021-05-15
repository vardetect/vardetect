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
import glob
from os.path import expanduser
import os, matplotlib
import scipy.misc
import numpy as np
import tensorflow as tf
from cfg import cfg, config
from IPython.display import clear_output, Image, display, HTML
import numpy as np    
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import FancyArrowPatch

class AnnotationHandler(HandlerLine2D):
    "Source: https://stackoverflow.com/questions/49261229/matplotlib-legend-unicode-marker"
    def __init__(self,ms,*args,**kwargs):
        self.ms = ms
        HandlerLine2D.__init__(self,*args,**kwargs)
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        ydata = ((height - ydescent) / 2.) * np.ones(xdata.shape, float)
        legline = FancyArrowPatch(posA=(xdata[0],ydata[0]),
                                  posB=(xdata[-1],ydata[-1]),
                                  mutation_scale=self.ms,
                                  **orig_handle.arrowprops)
        legline.set_transform(trans)
        return legline,

def create_dirs(dirs):
    
    if type(dirs) is not list:
        dirs = [dirs]
    
    for d in dirs:
        if not os.path.exists(d):
            print "creating dir: ", d 
            os.makedirs(d)  

            
def one_hot_labels(Y, dim):
    b = np.zeros((len(Y), dim))
    b[np.arange(len(Y)), Y] = 1

    return b            


def shuffle_data(X, Y):
    
    assert( len(X) == len(Y) ) , "Shuffling failed len(X) != len(Y)"
    
    perm = np.arange( len(X) )     
    np.random.seed(cfg.seed)
    np.random.shuffle(perm)
    
    X    = X[perm]
    Y    = Y[perm] 
    
    return X, Y

                        
def merge(images, size, sep_pixels=0):
    h, w, c = images.shape[1], images.shape[2], images.shape[3]
    img = np.ones((h * size[0], w * size[1] + sep_pixels * (size[1] - 1), c))
    
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*(w+sep_pixels):i*(w+sep_pixels)+(w), :] = image

    return img

def plot(img, plt, figsize=(3,3) ):
    plt.figure(figsize=figsize)
    if img.shape[2]== 1:
        plt.imshow(np.squeeze(img), cmap='gray')
    else:
        plt.imshow(img)    
    
    plt.show()
    
def append_zero_column( Y, dtype=np.float32 ):
    
    temp = np.zeros( (Y.shape[0], Y.shape[1]+1), dtype=dtype )
    temp[:,:-1] = Y
    
    return temp     
    
def insert_column( Y, pos='end', digit=0, dtype=np.float32 ):
    
    temp = np.ones( (Y.shape[0], Y.shape[1]+1), dtype=dtype ) * digit
    if pos == 'end':
        temp[:,:-1] = Y
    else:
        temp[:,1:]  = Y
        
    return temp     
  
    
def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)    
    
def get_accept_reject_ratio_samples(x, delta):
    
    num_samples = len(x)
    
    num_accept = np.sum( x < delta )
    num_reject = len(x) - num_accept
    
    reject_ratio = np.round(num_reject/float(num_samples)*100.0, 2 )
    
    accept_ratio = np.round(num_accept/float(num_samples)*100.0, 2 )
    
    return reject_ratio, accept_ratio, num_reject, num_accept
    
def plot_original_grid(og, plt):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    img_grid_og      = merge( og, [10,10] )             
    
    print img_grid_og.shape
    
    if img_grid_og.shape[2] == 1:
        ax.imshow(np.squeeze(img_grid_og), cmap = 'gray' ) 
    else:
        ax.imshow(img_grid_og ) 
   
    ax.set_axis_off()
    plt.show()    
    
    
def plot_original_decoded_grids(og, decoded, plt, save_path=None, prefix=None):
    
    n  = int( np.sqrt( len(og) ) )
    
    fig, ax = plt.subplots(1,2, figsize=(n,n))
    img_grid_og      = merge( og, [n,n] )             
    img_grid_decoded = merge( decoded, [n,n] )        
    
    
    if img_grid_og.shape[2] == 1:
        ax[0].imshow(np.squeeze(img_grid_og), cmap = 'gray' ) 
    else:
        ax[0].imshow(img_grid_og ) 
    
    if img_grid_decoded.shape[2] == 1:
        ax[1].imshow( np.squeeze(img_grid_decoded), cmap = 'gray' )
    else:        
        ax[1].imshow(img_grid_decoded )
   
    map(lambda axi: axi.set_axis_off(), ax.ravel())

        
    if save_path is not None:
        create_dirs([save_path])
        fpathpdf     = os.path.join( save_path, '{}_og_decoded_grid.pdf' .format(prefix) )
        fig.savefig(fpathpdf, bbox_inches='tight')
        
    plt.show()
    
    
def plot_latex_grid( img, sep_pixels=0, path=None):
    
    fig, ax = plt.subplots(1,1, figsize=(6,12))
    
    num_images = len(img)
        
    img_grid      = merge( img, [1,num_images], sep_pixels)             
        
    if img_grid.shape[2] == 1:
        ax.imshow(np.squeeze(img_grid), cmap = 'gray' ) 
    else:
        ax.imshow( img_grid ) 
        
    ax.set_axis_off()
    
    if path is not None:
        fig.savefig(path, bbox_inches='tight')
    
    plt.show()    
    
    return img_grid
    
    
    
def plot_pca( full_data, labels, num_samples=1000, markers=['o', 'x'], facecolors=[False, True], path=None, n_comp =3 ):
    
    color_maps = { 'Pos'   : 'dodgerblue',
                   'Neg'   : 'darkviolet',
                   'Neg-m' : 'darkgreen',
                   'ImageNet'  : 'orangered',
                   'ImageNet-m': 'darkgreen',
                   'Uniform'   : 'firebrick',
                   'Uniform-m' : 'darkgreen',
                   'CIFAR-10'  : 'orange',
                   'JbDA seed' : 'darkgreen',
                   'JbDA aug'  : 'tomato',
                   'JbDA aug-m': 'darkgreen',
                   'JbDA-T-RND-IFGSM seed' : 'darkgreen',
                   'JbDA-T-RND-IFGSM aug'  : 'tomato',
                   'JbDA-N-IFGSM seed' : 'darkgreen',
                   'JbDA-N-IFGSM aug'  : 'tomato',
                   'COLOR seed' : 'darkgreen',
                   'COLOR aug'  : 'tomato',
                   'Affnist'    : 'affnist',
                   'AltPD'  : 'darkgreen'
                 }
    
    project = PCA(n_components=n_comp)
    
    data = []
    
    n_ds = len(full_data)
    
    for i in range( n_ds ):
        data.append( full_data[i][:num_samples] )
                         
    full_zs = np.concatenate( data, axis = 0 )

    project.fit( full_zs  )
 
    colors= []

    fcs   = []
    
    points = dict()
    
    for color_map in color_maps.keys():
        points[color_map] = []
    
    for i in range( n_ds ):
        points[labels[i]].append(project.transform(data[i]))
        
        c = matplotlib.colors.cnames.get( color_maps.get(labels[i]) )
        colors.append( c )
         
        if facecolors[i]:
            fcs.append( c )
        else:
            fcs.append( 'none' )

    for color_map in color_maps.keys():
        if len(points[color_map]) == 0:
            del points[color_map]
        else:
            points[color_map] = np.squeeze(np.array(points[color_map]))
            
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if n_comp > 2:
        ax = Axes3D(fig)

    
    for i in range(n_ds):
        ax.scatter(*np.transpose(project.transform( data[i] )), marker=markers[i], color=colors[i], s=25, label=labels[i], facecolors=fcs[i]) 

    ax.legend(loc='best', numpoints=1, ncol=3, fontsize=8)    
            
    fname = []

    for i in range( len(labels) ):
        x = labels[i].lower().split(' ')[0]
        if x not in fname:
            fname.append(x)

    fname = '_'.join(fname)
        
    create_dirs( [path] )
        
    if path is not None:
        save_path = os.path.join(path, fname + '.pdf') 
        fig.savefig(save_path, bbox_inches='tight')            
    
    plt.show()
    
    for key in points:
        print os.path.join(path, fname + '-' + key.strip().split(' ')[-1].lower() + '.txt')
        np.savetxt(os.path.join(path, fname + '-' + key.strip().split(' ')[-1].lower() + '.txt'), points[key], fmt='%g')
    
    return points, fname


def plot_og_recon( og, recon, save_path, prefix, num_samples=10, sep_pixels=5):
    
    create_dirs([save_path])
    
    og_pdf      = os.path.join( save_path, '{}_og.pdf' .format(prefix) )

    recon_pdf = os.path.join( save_path, '{}_recon.pdf' .format(prefix) )

    delta_pdf = os.path.join( save_path, '{}_delta.pdf' .format(prefix) )

    _ = plot_latex_grid( og[-num_samples:], sep_pixels=sep_pixels, path=og_pdf )

    _ = plot_latex_grid( recon[-num_samples:], sep_pixels=sep_pixels, path=recon_pdf )

    delta = np.abs( og - recon )

    _ = plot_latex_grid( delta[-num_samples:],sep_pixels=5, path=delta_pdf )
    
    
def print_dist_params( x, nth_percentile = 99 ):
    print "mean: {} std: {} var: {} min: {} max: {} {}-percentile: {}" .format(  np.mean(x), np.std(x), np.var(x), np.min(x), np.max(x), nth_percentile, np.percentile(x, nth_percentile )  )    
    
    
def plot_recon_histograms(rlosses_true_train, rlosses_true_val, rlosses_true_test, rlosses_unif_noise, rlosses_pap_seed, rlosses_pap_aug, rlosses_cifar_noise, rlosses_img_noise, save_path, bins=30, ylim_max=4000, step_size=10):
       
    create_dirs(save_path)
        
    fig, ax = plt.subplots(6,1, figsize=(8,12) )

    percentile = 99

    print "Validation info:"

    print_dist_params(rlosses_true_val)
    val_threshold = float( np.percentile(rlosses_true_val, percentile) ) 

    print "Training info"

    train_reject_ratio, train_accept_ratio, train_num_reject, train_num_accept = get_accept_reject_ratio_samples(rlosses_true_train, val_threshold)

    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( train_accept_ratio, train_num_accept, len(rlosses_true_train), train_reject_ratio )

    np.savetxt( os.path.join(save_path, 'train.txt' ) , rlosses_true_train, fmt='%g')
    
    print "val threshold is set to {} for {} percentile" .format( val_threshold, percentile )

    val_reject_ratio, val_accept_ratio, val_num_reject, val_num_accept = get_accept_reject_ratio_samples(rlosses_true_val, val_threshold)
    
    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( val_accept_ratio, val_num_accept, len(rlosses_true_val), val_reject_ratio )

    np.savetxt( os.path.join(save_path, 'val.txt' ) , rlosses_true_val, fmt='%g')
    
    ax[0].hist(rlosses_true_val, bins=bins, color='lightskyblue')    

    print "\nTest info:"
    print_dist_params(rlosses_true_test)
    test_reject_ratio, test_accept_ratio, test_num_reject, test_num_accept = get_accept_reject_ratio_samples(rlosses_true_test, val_threshold)

    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( test_accept_ratio, test_num_accept, len(rlosses_true_test), test_reject_ratio )
        
    np.savetxt( os.path.join(save_path, 'test.txt' ) , rlosses_true_test, fmt='%g')

    ax[1].hist(rlosses_true_test, bins=bins, color = 'g')    

    print "\nUnifNoise info:"
    print_dist_params(rlosses_unif_noise)

    unif_reject_ratio, unif_accept_ratio, unif_num_reject, unif_num_accept = get_accept_reject_ratio_samples(rlosses_unif_noise, val_threshold)

    np.savetxt( os.path.join(save_path, 'unif.txt' ) , rlosses_unif_noise, fmt='%g')
    
    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( unif_accept_ratio, unif_num_accept, len(rlosses_unif_noise), unif_reject_ratio )
    ax[2].hist(rlosses_unif_noise, bins=100, color = 'firebrick')    

    print "\nPap info:"
    rlosses_pap = np.concatenate( [rlosses_pap_seed, rlosses_pap_aug], axis=0)
    
    np.savetxt( os.path.join(save_path, 'jbda-aug.txt' ) , rlosses_pap_aug, fmt='%g')
    np.savetxt( os.path.join(save_path, 'jbda-seed.txt' ) , rlosses_pap_seed, fmt='%g')
    
    print_dist_params(rlosses_pap)

    pap_reject_ratio, pap_accept_ratio, pap_num_reject, pap_num_accept = get_accept_reject_ratio_samples(rlosses_pap, val_threshold)

    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( pap_accept_ratio, pap_num_accept, len(rlosses_pap_aug), pap_reject_ratio )
    ax[3].hist(rlosses_pap_seed, bins=5, color = 'g')   
    ax[3].hist(rlosses_pap_aug, bins=bins, color = 'tomato')   

    print "\nCifarNoise info:"
    print_dist_params(rlosses_cifar_noise)

    cifar_reject_ratio, cifar_accept_ratio, cifar_num_reject, cifar_num_accept = get_accept_reject_ratio_samples(rlosses_cifar_noise, val_threshold)

    np.savetxt( os.path.join(save_path, 'cifar.txt' ) , rlosses_cifar_noise, fmt='%g')
    
    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( cifar_accept_ratio, cifar_num_reject, len(rlosses_cifar_noise), cifar_reject_ratio )

    ax[4].hist(rlosses_cifar_noise, bins=bins, color = 'orange')    

    print "\nImgNoise info:"
    print_dist_params(rlosses_img_noise)

    img_reject_ratio, img_accept_ratio, img_num_reject, img_num_accept = get_accept_reject_ratio_samples(rlosses_img_noise, val_threshold)
    
    np.savetxt( os.path.join(save_path, 'imagenet.txt' ) , rlosses_img_noise, fmt='%g')

    print "Acceptance(%):{}={}/{} Rejection(%)={}" .format( img_accept_ratio, img_num_accept , len(rlosses_img_noise), img_reject_ratio )

    ax[5].hist(rlosses_img_noise, bins=bins, color = 'orangered')    

    for i in range(6):
        ax[i].axvline(val_threshold, color='k', linestyle='dashed', linewidth=1)
        
#         if i!=5:
#             ax[i].set_axis_off()
#             ax[i].tick_params(axis='both', which='both', length=0)
#             ax[i].tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

#         a1 = ax[i].annotate("", xy=(val_threshold+1, ylim_max-500), xycoords='data', xytext=(val_threshold-step_size, ylim_max-500), textcoords='data', arrowprops=dict(arrowstyle="<-",
#                     connectionstyle="arc3", color='forestgreen'), label='Inlier' )

#         a2= ax[i].annotate("", xy=(val_threshold+step_size, ylim_max-500), xycoords='data', xytext=(val_threshold, ylim_max-500), textcoords='data', arrowprops=dict(arrowstyle="->",
#                     connectionstyle="arc3", color='crimson'), label='Outlier' )

#         h, l = ax[i].get_legend_handles_labels()
#         ax[i].legend(handles = h +[a1,a2], handler_map={type(a1) : AnnotationHandler(5)})

    rlosses_ = np.concatenate([rlosses_true_val, rlosses_true_test, rlosses_unif_noise, rlosses_pap_seed, rlosses_pap_aug, rlosses_cifar_noise, rlosses_img_noise ], axis=0)    

    # Setting the values for all axes.
    xlim = (0, np.max(rlosses_))
    ylim = (0, ylim_max)
    plt.setp(ax, xlim=xlim, ylim=ylim)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
    
    plt.show()


    
    for i in range(6):
        extent = ax[i].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(os.path.join( save_path, 'ax-%d.pdf' % i), bbox_inches=extent)   
                        
    rr = [ train_reject_ratio, val_reject_ratio, test_reject_ratio, unif_reject_ratio, pap_reject_ratio, cifar_reject_ratio, img_reject_ratio]     
        
    print r'\multicolumn{1}{l}{F-MNIST} & ' + ' & '.join( map(str, rr) ) + r'\\'
            
    
    
def fetch_filename_jbda(og_net, tr_dataset, cp_net, jtype, budget, eps , defense_type=None, path=None, extract=True, filtered=False):

    if path is None:
#         basepath = os.path.join( expanduser('~'), 'copynet_defense_cygnus', 'logdir', 'train_logs', 'onehot')
        basepath = os.path.join( expanduser('~'), 'projects', 'copynet_defense', 'logdir', 'train_logs', 'onehot')
    
    fname = '_'.join([og_net, tr_dataset, cp_net, tr_dataset,'{}-{}' .format(jtype, eps) , str(budget)])
    
    if defense_type is not None:
        fname = '{}_{}' .format( fname, defense_type  )        
    
    
    if extract:
        filename =  fname + '.log_{}_jbda' .format( 'extract_model' )
    else:
        if (defense_type is None) or filtered:
            filename =  fname + '.log_{}_jbda' .format( 'attack' )
        else:
            filename =  fname + '.log_{}_jbda_{}' .format( 'attack', 'unfiltered' )


            
    filepath = os.path.join(basepath, filename)
    
    matched_filenames = glob.glob(filepath)
    
    
    if len(matched_filenames) == 0:
        return 'NOTFOUND'

    assert len(matched_filenames) == 1, 'Matched filenames: ' + '; '.join(matched_filenames) + ' by: ' + filepath
            
    return matched_filenames[0]





def fetch_filename_tramer(og_net, tr_dataset, cp_net, method, budget, defense_type=None, path=None, extract=True, filtered=False):

    if path is None:
#         basepath = os.path.join( expanduser('~'), 'copynet_defense_cygnus', 'logdir', 'train_logs', 'onehot')
        basepath = os.path.join( expanduser('~'), 'projects', 'copynet_defense', 'logdir', 'train_logs', 'onehot')
    
    fname = '_'.join([og_net, tr_dataset, cp_net, 'uniform', method , str(budget)])
    
    if defense_type is not None:
        fname = '{}_{}' .format( fname, defense_type  )        
        
    if extract:
        filename =  fname + '.log_{}_tramer' .format( 'extract_model' )
    else:
        if (defense_type is None) or filtered:
            filename =  fname + '.log_{}_tramer' .format( 'attack' )
        else:
            filename =  fname + '.log_{}_tramer_{}' .format( 'attack', 'unfiltered' )
         
    filepath = os.path.join(basepath, filename)
    
    matched_filenames = glob.glob(filepath)
    
    if len(matched_filenames) == 0:
        return 'NOTFOUND'

    assert len(matched_filenames) == 1, 'Matched filenames: ' + '; '.join(matched_filenames) + ' by: ' + filepath
            
    return matched_filenames[0]



def fetch_filename(og_net, tr_dataset, cp_net, ns_dataset, strategy, budget, defense_type=None, num_iter=10, path=None, extract=True, filtered=False):
    
    if path is None:
#         basepath = os.path.join( expanduser('~'), 'copynet_defense_cygnus', 'logdir', 'train_logs', 'onehot')
        basepath = os.path.join( expanduser('~'), 'projects', 'copynet_defense', 'logdir', 'train_logs', 'onehot')
    
    if strategy!='random':
        k            = int( (0.7*budget)/num_iter )
        initial_seed = int(0.1*budget)    
        val_size     = int(0.2*budget)
    else:
        k            = 0
        initial_seed = int(0.8*budget)    
        val_size     = int(0.2*budget)        
    
    suffix = '{}+{}+{}' .format( initial_seed, val_size , num_iter * k  ) 
    
    if defense_type is not None:
        suffix = '{}_{}' .format( suffix, defense_type  )
    
    if extract:
        ext = '.log_{}_activethief' .format( 'extract_model' )
    else:
        if (defense_type is None) or filtered:
            ext = '.log_attack_activethief' 
        else:
            ext = '.log_attack_activethief_unfiltered' 
            
    filename = '_'.join([og_net, tr_dataset, cp_net, ns_dataset, strategy, suffix]) + ext
    filepath = os.path.join(basepath, filename)
    
    matched_filenames = glob.glob(filepath)
    
    if len(matched_filenames) == 0:
        return 'NOTFOUND'

    assert len(matched_filenames) == 1, 'Matched filenames: ' + '; '.join(matched_filenames) + ' by: ' + filepath
    
    if defense_type is None:
        budget_string = matched_filenames[0].split('/')[-1].split('.')[0].split('_')[-1]
    else:
        budget_string = matched_filenames[0].split('/')[-1].split('.')[0].split('_')[-4]
        
    if type(budget) == int:
        assert eval(budget_string) == budget
    
    return matched_filenames[0]
        
def base_kernel(x, y, sigma):
    norm_square = np.linalg.norm(x-y) ** 2
    sigma_square = sigma ** 2
    
    return np.exp(- norm_square /(2* sigma_square))

def composite_kernel(x, y, sigmas):
    result = 0
    
    for sigma in sigmas:
        result += base_kernel(x, y, sigma)
        
    return result


def compute_mmd(dataset_x, dataset_y, sigmas=[1, 5, 10, 15, 20]):
    result = 0
    
    n = len(dataset_x)
    m = len(dataset_y)
    
    for i in range(n):
        for j in range(n):
            result += 1./(n**2) * composite_kernel(dataset_x[i], dataset_x[j], sigmas)
    
    for i in range(n):
        for j in range(m):
            result -= 2./(n*m) * composite_kernel(dataset_x[i], dataset_y[j], sigmas)
    
    for i in range(m):
        for j in range(m):
            result += 1./(m**2) * composite_kernel(dataset_y[i], dataset_y[j], sigmas)
            
    return np.sqrt(result)

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def.
       #Orginal Source:               https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
       #Modified Souce: https://stackoverflow.com/questions/41388673/visualizing-a-tensorflow-graph-in-jupyter-doesnt-work/41463991#41463991"""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def



def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph.
       #Orginal Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
       #Modified Souce: https://stackoverflow.com/questions/41388673/visualizing-a-tensorflow-graph-in-jupyter-doesnt-work/41463991#41463991"""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))    
