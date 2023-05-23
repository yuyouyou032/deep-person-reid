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


class Til2023CvDataset(ImageDataset):
    dataset_dir = 'til2023_cv_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        path = "/notebooks/deep-person-reid/reid-data/til2023_cv_dataset/Train1/"
        paths = {}
        # Iterate over the files in the directory
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                if filename[:5] not in paths.keys():
                    paths[filename[:5]] = [filename]
                else:
                    paths[filename[:5]].append(filename)
                
        paths.pop('.DS_S')
        train = []
        query = []
        gallery = []
        
        key_list = list(paths.keys())

        for i in key_list:
            a = paths[i]
            shuffle(a)
            for j in range(len(a)):
                if j<30:
                    train.append((path + a[j], int(a[j][:5]), 0))
                elif j>=30 and j<41:
                    query.append((path + a[j], int(a[j][:5]), 0))
                else:
                    gallery.append((path + a[j], int(a[j][:5]), 2)) #camera for gallery has to be different, i think
        print(train[:10])
        print(query[:10])
        print(gallery[:10])

        super(Til2023CvDataset, self).__init__(train, query, gallery, **kwargs)