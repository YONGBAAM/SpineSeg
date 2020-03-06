import torch

import numpy as np
from label_io import read_data_names, read_images, read_segmaps, read_labels

from dataset import SegDataset

train_data_names = read_data_names('./train_labels')
train_segmaps = read_segmaps('./train_labels', train_data_names)

train_dataset = SegDataset('./train_images', train_segmaps, train_data_names, None)

for train_d in train_dataset:
    i = train_d['image']
    l = train_d['label']

    print(i.shape)
    print(torch.mean(i))
