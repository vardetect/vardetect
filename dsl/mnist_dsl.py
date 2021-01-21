
import numpy as np, os, struct
from base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join
from cfg import cfg

class MNISTDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=False, seed=1337, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, sample_limit=None, keep_class=None, num_classes=10):

        if mode == 'val':
            assert val_frac is not None

        if path is None:
            self.path = os.path.join(cfg.home, 'datasets', 'mnist')
        else:
            self.path = path
        
        super(MNISTDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            keep_class = keep_class,
            num_classes = num_classes
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        if mode == 'test':
            fname_img = os.path.join(self.path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 't10k-labels-idx1-ubyte')
        else:
            assert mode == 'train' or mode == 'val'
            fname_img = os.path.join(self.path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(self.path, 'train-labels-idx1-ubyte')

        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.labels = np.fromfile(flbl, dtype=np.int8)
        
        with open(fname_img, 'rb') as fimg:
            print fname_img
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.data = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.labels), rows, cols, 1)
                
        print self.data.shape
        print self.labels.shape
        
        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
        
        if self.keep_class is not None:
            class_data = []
            class_labels = []
            for i, x in enumerate(self.data):
                if int( self.labels[i] ) in self.keep_class:
                    class_data.append(x)
                    class_labels.append(self.labels[i])
            
            self.data   = np.array(class_data)            
            self.labels = np.array(self.labels, dtype=np.int32)    
            
        self.labels = np.squeeze(self.labels)


