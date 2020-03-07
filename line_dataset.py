import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os.path import join as osj

from label_io import read_labels, read_data_names, plot_image, hwc, chw, read_segmaps

# from label_transform import CoordCustomPad, CoordHorizontalFlip,CoordRandomRotate, CoordLabelNormalize, CoordResize, CoordVerticalFlip

val_ratio = 0.1

from segtransforms import SegResize


###############################################
#
#       Refactoring Finished
#
###############################################

class LineDataset(Dataset):
    def __init__(self, segmap_location, line_location, segmap_names, line_names, transform_list = None):
        super(LineDataset).__init__()
        self.segmap_location = segmap_location
        self.line_location = line_location
        self.segmap_names = segmap_names
        self.line_names = line_names

        self.size = len(segmap_names)
        self.transform_list = transform_list

        self.toTensor = transforms.ToTensor()
        self.nor = transforms.Normalize((0.5,), (0.5,))
        self.toImage = transforms.ToPILImage()

        if len(segmap_names) != len(line_names):
            assert ('segmaps and lines len do not match {},{}'.format(len(segmap_names), len(line_names)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        segmap = np.load(osj(self.segmap_location, self.segmap_names[idx]))
        line = np.load(osj(self.line_location, self.line_names[idx]))

        segmap = self.toImage(np.uint8(segmap*255))
        line = self.toImage(np.uint8(line*255))

        if self.transform_list != None:
            for transform in self.transform_list:
                segmap, line = transform(segmap, line)

        segmap = self.toTensor(segmap)
        line = self.toTensor(line)
        return dict(image=segmap, label=line)

def get_lineloader_train_val(batch_size_tr, batch_size_val = 1, shuffle = True):
    segmap_path = './train_labels'
    line_path = './train_lines'

    segmap_names = read_data_names(segmap_path, 'segmap_names')
    line_names = read_data_names(line_path, 'line_names')

    N_all = len(segmap_names)
    N_val = int(N_all * val_ratio)
    N_train = N_all - N_val
    # get train and validation data set

    segmap_names_train = []
    segmap_names_val = []
    line_names_train = []
    line_names_val = []

    transform_list = [SegResize((512,256))]

    if not os.path.exists(os.path.join('./', 'val_permutation.npy')):
        print('reset permutation')
        permutation = np.random.permutation(N_all)
        np.save(os.path.join('./', 'val_permutation.npy'), permutation)
    else:
        permutation = np.load(os.path.join('./', 'val_permutation.npy'))

    for ind in permutation[:N_train]:
        segmap_names_train.append(segmap_names[ind])
        line_names_train.append(line_names[ind])

    for ind in permutation[N_train:]:
        segmap_names_val.append(segmap_names[ind])
        line_names_val.append(line_names[ind])

    #########################
    dset_train = LineDataset(segmap_location = segmap_path, line_location = line_path,
                             segmap_names = segmap_names_train, line_names = line_names_train,
                             transform_list = transform_list)
    dset_val = LineDataset(segmap_location = segmap_path, line_location = line_path,
                             segmap_names = segmap_names_val, line_names = line_names_val,
                             transform_list = transform_list)

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size_tr, shuffle=shuffle)
    loader_val = DataLoader(dataset=dset_val, batch_size=batch_size_val, shuffle=False)
    return loader_train, loader_val

