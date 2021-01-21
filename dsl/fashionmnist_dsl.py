import numpy as np, os, struct
from mnist_dsl import MNISTDSL
from os.path import expanduser, join

class FashionMNISTDSL(MNISTDSL):
    def __init__(self, batch_size, shuffle_each_epoch=True, seed=1337, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, keep_class=None, sample_limit=None):

        self.my_dict = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',  8: 'Bag', 9: 'Ankle boot' }
        
        if path is None:
            path = os.path.join('datasets', 'fashionmnist')
        
        super(FashionMNISTDSL, self).__init__(
            batch_size,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            path=path,
            resize=resize,
            keep_class = keep_class,
        )

        
        
    def get_descripiton( self, preds ):
        return np.vectorize(self.my_dict.get)(preds)
        