import numpy as np, os, pandas as pd, cv2
from base_dsl import BaseDSL, one_hot_labels
from os.path import expanduser, join
import cv2
from skimage import color, exposure, transform
from cfg import cfg

def preprocess_img(img, resize):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2,
              :]

    # rescale to standard size
    img = transform.resize(img, resize)

    return img

class GTSRDSL(BaseDSL):
    def __init__(self, batch_size, shuffle_each_epoch=True, seed=1337, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=(32,32), keep_class=None, preprocess=False, sample_limit=None, num_classes=43):
        
        if mode == 'val':
            assert val_frac is not None

        if resize is None:
            resize = (32,32)
        assert type(resize) == tuple and len(resize) == 2 and type(resize[0]) == int and type(resize[1]) == int

        self.resize     = resize
        self.preprocess = preprocess


        if path is None:
            home = cfg.home
            self.path = os.path.join(home, 'datasets', 'gtsrb')
        else:
            self.path = path
        
        super(GTSRDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            keep_class = keep_class,
            sample_limit = sample_limit,
            num_classes  = num_classes
        )
        
    def is_multilabel(self):
        return False
    

    def load_data(self, mode, val_frac):
        
        if self.preprocess:        
            if mode == 'train' or mode == 'val':
                base_path = os.path.join(self.path, 'Final_Training', 'Images')

                self.data           = []
                self.labels         = []

                for c in range(self.num_classes): 
                    x = [os.path.join('%05d' % c, f) for f in os.listdir(os.path.join(base_path, '%05d' % c)) if f.endswith('.ppm')]
                    y = np.ones(len(x),dtype=np.int32) * c
                    self.data.extend(x)
                    self.labels.append(y)

                self.data   = np.array(self.data)
                self.labels = np.concatenate(self.labels, axis=0)
            else:
                base_path = os.path.join(self.path, 'Final_Test', 'Images')

                df = pd.read_csv(os.path.join(self.path, 'Final_Test', 'GT-final_test.csv'), delimiter=';')
                self.data = np.array(df['Filename'])
                self.labels = np.array(df['ClassId'])

            labels = self.labels             

            # Read in data from files
            data = []

            for i, image_path in enumerate(self.data):
                image_full_path = os.path.join(base_path, image_path)
                img             = cv2.imread(image_full_path, cv2.IMREAD_COLOR)   
                img             = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img             = preprocess_img(img, self.resize)
                data.append(img)
        else:
            if  mode == 'train' or mode == 'val':
                data   = np.load(os.path.join(self.path, 'train-images.npy') )
                labels = np.load(os.path.join(self.path, 'train-labels.npy') )                           
            else:
                data   = np.load( os.path.join(self.path, 'test-images.npy') )
                labels = np.load( os.path.join(self.path, 'test-labels.npy') )                                                             
        self.data   = data
        self.labels = labels 
                        
        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)        
                    
        self.labels = np.squeeze(self.labels)
                
        if self.keep_class is not None:
            class_data = []
            class_labels = []
            for i, x in enumerate(self.data):
                if int( self.labels[i] ) in self.keep_class:
                    class_data.append(x)
                    class_labels.append(self.labels[i])
            
            self.data   = np.array(class_data)            
            self.labels = np.array(self.labels, dtype=np.int32)
        
        if self.sample_limit is not None:
            self.data   = self.data[:self.sample_limit]
            self.labels = self.labels[:self.sample_limit]        
