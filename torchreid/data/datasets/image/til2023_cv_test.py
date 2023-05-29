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


class Til2023CvTest(ImageDataset):
    dataset_dir = 'til2023_cv_test'
    

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        path = "/notebooks/deep-person-reid/reid-data/til2023_cv_dataset/Train1/"
        path2 = "/notebooks/deep-person-reid/reid-data/til2023_cv_test/Suspects/"
        path3 = "/notebooks/deep-person-reid/reid-data/til2023_cv_test/Test1/"
        paths = {}
        
        # prepare train set
        train = []
        # Iterate over the files in the directory
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                if filename[:5] not in paths.keys():
                    paths[filename[:5]] = [filename]
                else:
                    paths[filename[:5]].append(filename)   
        key_list = list(paths.keys())
        for i in key_list:
            a = paths[i]
            shuffle(a)
            for j in range(len(a)):
                train.append((path + a[j], int(a[j][:5]), 0))
                
                
          
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


        super(Til2023CvTest, self).__init__(train, query, gallery, **kwargs)
        