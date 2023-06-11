from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp

from PIL import Image
from random import shuffle
import random
random_seed = 42
random.seed(random_seed)

from torchreid.data import ImageDataset


class Robo(ImageDataset):
    dataset_dir = 'robo'
    

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        path = "/notebooks/deep-person-reid/reid-data/til2023_cv_dataset/Train1/"

        path2 = "/notebooks/deep-person-reid/reid-data/sus/"
        
        path3 = "/notebooks/deep-person-reid/reid-data/test/"
        
        paths = {}
        
        # prepare train set
        train = [("",0,0)]         
          
        # prepare query and gallery
        query = []
        gallery = []
        # Iterate over the files in the directory
        for filename in os.listdir(path2):
            query.append((path2 + filename, 0, 0))

                    
        for filename in os.listdir(path3):           
            gallery.append((path3 + filename, 0, 2))
                
        
        shuffle(train)
        shuffle(query)
        shuffle(gallery)
                
        print("TRAIN::: ", train[:10])
        print("QUERY::: ", query[:10])
        print("GALLERY::: ", gallery[:10])


        super(Robo, self).__init__(train, query, gallery, **kwargs)
        