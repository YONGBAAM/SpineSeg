import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from label_io import read_labels, read_data_names, plot_image, hwc, chw, read_segmaps
from segtransforms import SegResize
from os.path import join as osj

# from label_transform import CoordCustomPad, CoordHorizontalFlip,CoordRandomRotate, CoordLabelNormalize, CoordResize, CoordVerticalFlip

val_ratio = 0.1

from segtransforms import SegResize
###############################################
#
#       Refactoring Finished
#
###############################################

class SegDataset(Dataset):
    def __init__(self, data_location, segmap_location, data_names, segmap_names, transform_list = None):
        super(SegDataset).__init__()
        self.data_location = data_location
        self.segmap_location = segmap_location
        self.data_names = data_names
        self.segmap_names = segmap_names
        self.size = len(data_names)
        self.transform_list = transform_list
        self.toTensor = transforms.ToTensor()
        self.nor = transforms.Normalize((0.5,), (0.5,))
        self.toImage = transforms.ToPILImage()



    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_location, self.data_names[idx])
        image = Image.open(img_path)

        label_np = np.load(osj(self.segmap_location, self.segmap_names[idx]))
        label = self.toImage(np.uint8(label_np*255))
        # label = hwc(label)
        # #label도 똑같이 PIL image로
        # label = Image.fromarray(np.uint8(label*255))

        #이상태 755 2125

        if self.transform_list != None:
            for transform in self.transform_list:
                image, label = transform(image, label)

        image = self.toTensor(image)
        if image.shape[0] ==3:
            image = torch.mean(image, dim = 0, keepdim = True)
        image = self.nor(image)

        label = self.toTensor(label)
        #맞네?
        return {'image' : image, 'label' : label}

def get_loader_train_val(batch_size_tr=64, batch_size_val=1, shuffle = True):
    data_path = './train_images'
    label_path = './train_labels'

    val_ratio = 0.1
    transform_list = [SegResize((512, 256))]

    data_names = read_data_names(label_path)
    segmap_names = read_data_names(label_path, title = 'segmap_names')

    N_all = len(data_names)
    N_val = int(N_all * val_ratio)
    N_train = N_all - N_val
    # get train and validation data set
    data_names_train = []
    data_names_val = []
    segmap_names_train = []
    segmap_names_val = []

    if not os.path.exists(os.path.join('./', 'val_permutation.npy')):
        print('reset permutation')
        permutation = np.random.permutation(N_all)
        np.save(os.path.join('./', 'val_permutation.npy'), permutation)
    else:
        permutation = np.load(os.path.join('./', 'val_permutation.npy'))

    for ind in permutation[:N_train]:
        data_names_train.append(data_names[ind])
        segmap_names_train.append(segmap_names[ind])

    for ind in permutation[N_train:]:
        data_names_val.append(data_names[ind])
        segmap_names_val.append(segmap_names[ind])

    #########################
    dset_train = SegDataset(data_location=data_path, segmap_location=label_path,
                            data_names=data_names_train, segmap_names=segmap_names_train, transform_list=transform_list)
    dset_val = SegDataset(data_location=data_path, segmap_location=label_path,
                            data_names=data_names_val, segmap_names=segmap_names_val, transform_list=transform_list)

    loader_train = DataLoader(dataset=dset_train, batch_size=batch_size_tr, shuffle=shuffle)
    loader_val = DataLoader(dataset=dset_val, batch_size=batch_size_val, shuffle=False)
    return loader_train, loader_val


rr = []
cc = []
for r in range(512):
    for c in range(256):
        if r % 20 == 0 and c %20 ==0:
            rr.append(r)
            cc.append(c)
            # sample_indices.append((0,r,c))

rlen = 26
clen = 13



if __name__ == '__main__':
    author = 'YB'
    #display data image
    import pandas as pd

    def get_arr_sample(arr,H=512,W=256):
        res = arr.reshape(512,256)[rr,cc].reshape(rlen,clen)
        return res


    tl,vl = get_loader_train_val(24)

    td = next(iter(tl))
    images = td['image'].detach().cpu().numpy()
    segmaps = td['label'].detach().cpu().numpy()
    print(images.shape) #24 1 512 256
    print(segmaps.shape) #24 1 512 256

    for i in range(24):
        im = images[i]
        lb = segmaps[i]

        #denormalize image
        im = np.interp(im, (im.min(), im.max()), (0,1))

        all_im = im*0.8 + lb * 0.2

        sample = get_arr_sample(all_im)
        pd.DataFrame(sample).to_csv('./plots/{}.csv'.format(i), index=None, header=None)
        plt.figure()
        plt.imshow(hwc(all_im))
        plt.savefig('./plots/{}.png'.format(i))
        plt.close()
