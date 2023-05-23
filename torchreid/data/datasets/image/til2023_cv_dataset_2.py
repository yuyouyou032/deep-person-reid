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


class Til2023CvDataset2(ImageDataset):
    dataset_dir = 'til2023_cv_dataset'
    

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        path = "/notebooks/deep-person-reid/reid-data/til2023_cv_dataset/Train1/"
        path2 = "/notebooks/deep-person-reid/reid-data/til2023_cv_dataset/Validation1/"
        paths = {}
        paths2 = {}
        
        # prepare train set
        train = []
        # Iterate over the files in the directory
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                if filename[:5] not in paths.keys():
                    paths[filename[:5]] = [filename]
                else:
                    paths[filename[:5]].append(filename)   
        paths.pop('.DS_S')
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
            if os.path.isfile(os.path.join(path2, filename)):
                if filename[:5] not in paths2.keys():
                    paths2[filename[:5]] = [filename]
                else:
                    paths2[filename[:5]].append(filename)   
        paths2.pop('.DS_S')
        key_list2 = list(paths2.keys())
        for i in key_list2:
            a = paths2[i]
            shuffle(a)
            for j in range(len(a)):
                if 0<=j<5:
                    query.append((path2 + a[j], int(a[j][:5])+200, 0))
                else:
                    gallery.append((path2 + a[j], int(a[j][:5])+200, 2))
                
        
        shuffle(train)
        shuffle(query)
        shuffle(gallery)
                
        print("TRAIN::: ", train[:10])
        print("QUERY::: ", query[:10])
        print("GALLERY::: ", gallery[:10])


        super(Til2023CvDataset2, self).__init__(train, query, gallery, **kwargs)
        