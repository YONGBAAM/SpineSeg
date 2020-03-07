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
    def __init__(self, segmaps, lines, transform_list = None):
        super(LineDataset).__init__()
        self.segmaps = segmaps
        self.lines = lines
        self.size = len(segmaps)
        self.transform_list = transform_list
        self.toTensor = transforms.ToTensor()
        self.nor = transforms.Normalize((0.5,), (0.5,))
        self.toImage = transforms.ToPILImage()

        if len(segmaps) != len(lines):
            assert ('segmaps and lines len do not match {},{}'.format(len(segmaps), len(lines)))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        segmap = self.segmaps[idx]
        line = self.lines[idx]

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

    segmap_names = [n for n in os.listdir(segmap_path) if n[-4:] == '.npy']

    segmaps = [np.load(osj(segmap_path, n)) for n in segmap_names]
    lines = [np.load(osj(line_path, n)) for n in segmap_names]

    N_all = len(segmaps)
    N_val = int(N_all * val_ratio)
    N_train = N_all - N_val
    # get train and validation data set

    segmaps_train = []
    segmaps_val = []
    lines_train = []
    lines_val = []

    transform_list = [SegResize((512,256))]

    if not os.path.exists(os.path.join('./', 'val_permutation.npy')):
        print('reset permutation')
        permutation = np.random.permutation(N_all)
        np.save(os.path.join('./', 'val_permutation.npy'), permutation)
    else:
        permutation = np.load(os.path.join('./', 'val_permutation.npy'))

    for ind in permutation[:N_train]:
        segmaps_train.append(segmaps[ind])
        lines_train.append(lines[ind])
    segmaps_train = np.asarray(segmaps_train)
    lines_train = np.asarray(lines_train)

    for ind in permutation[N_train:]:
        segmaps_val.append(segmaps[ind])
        lines_val.append(lines[ind])
    segmaps_val = np.asarray(segmaps_val)
    lines_val = np.asarray(lines_val)

    #########################
    dset_train = LineDataset(segmaps_train, lines_train, transform_list=transform_list)
    dset_val = LineDataset(segmaps_val, lines_val, transform_list= transform_list)

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size_tr, shuffle=shuffle)
    loader_val = DataLoader(dataset=dset_val, batch_size=batch_size_val, shuffle=False)
    return loader_train, loader_val

