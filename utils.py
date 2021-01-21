import numpy as np, matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collect_aux_data(dictionary_list, key):
    result = []
    
    for item in dictionary_list:
        result.append(item[key])
            
    return np.array(result)

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

def get_metric(model, sess, dsl, metric):    
    preds = []
    num_batches = dsl.get_num_batches()
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

def compute_agreement_and_accuracy(true_model, copy_model, sess, test_dsl):
    copy_acc = compute_evaluation_measure(copy_model, sess, test_dsl, copy_model.sum_correct_prediction)                
    print "Test Set Accuracy: %.2f%%" % (copy_acc*100)
    
def sample_z(model, dsl, sess):
    return get_metric(model, sess, dsl, model.mean)


def plot_pca(full_data, labels, num_samples=1000, markers=['o', 'x'], facecolors=[False, True], n_comp =3 ):
    color_maps = {
        'Confidential': 'dodgerblue',
        'Outlier'     : 'darkviolet',
        'NPD'         : 'orangered',
        'Syn'         : 'firebrick',
        'AdvPD seed'  : 'darkgreen',
        'AdvPD'       : 'tomato',
        'AltPD'       : 'darkgreen',
        'PD'          : 'darkgreen'
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
        colors.append(c)
         
        if facecolors[i]:
            fcs.append(c)
        else:
            fcs.append('none')

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

    plt.show()

