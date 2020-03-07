import matplotlib.pyplot as plt

from skimage.draw import line, polygon, circle, circle_perimeter, ellipse
from PIL import Image
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps
import scipy.io as spio
import pandas as pd

from label_io import plot_image

from label_io import read_labels, read_images, read_data_names

def draw_seg(coord, H, W):
    seg_image = np.zeros((H, W, 3))
    coord_rev = coord.reshape(-1, 2, 2)
    pol = np.concatenate((coord_rev[0, 0, :].reshape(1, 2), coord_rev[:, 1, :], coord_rev[::-1, 0, :]), axis=0)
    rr, cc = polygon(pol[:, 1], pol[:, 0], seg_image.shape)
    seg_image[rr, cc, :] = 1
    seg_image = seg_image[:, :, 0]
    return seg_image
from PIL import ImageDraw
def draw_line(coord, H, W):
    line_image = np.zeros((H, W, 3))
    img = Image.new('RGB', (W,H))
    img_line = ImageDraw.Draw(img)

    coord_rev = coord.reshape(-1, 2, 2)
    coord_mid = np.mean(coord_rev, axis = 1).astype(int)
    for i in range(coord_mid.shape[0] -1):

        start_coord = coord_mid[i]
        end_coord = coord_mid[i+1]

        img_line.line(((start_coord[0], start_coord[1]), (end_coord[0],end_coord[1]))
                      , width = 20, fill = 'white')

    # img.show()
    np_image = np.asarray(img)
    # print(np_image.shape)
    np_image = np.mean(np_image, axis = 2)
    # line_image = line_image[:, :, 0]
    return np_image

def prepare_seg():
    train_data_names = read_data_names('./train_labels')
    test_data_names = read_data_names('./test_labels')

    train_images = read_images('./train_images', data_names=train_data_names)
    train_labels = read_labels('./train_labels')
    # train_data_names = read_data_names('./train_labels')

    test_labels = read_labels('./test_labels')
    test_images = read_images('./test_images', test_data_names)
    # test_data_names = read_data_names('./test_labels')

    for ind, image in enumerate(train_images):
        H, W, C = image.shape
        lab = train_labels[ind]
        seg = draw_seg(lab, H, W)
        np.save('./train_labels/{}.npy'.format(ind), seg)

    for ind, image in enumerate(test_images):
        H, W, C = image.shape
        lab = test_labels[ind]
        seg = draw_seg(lab, H, W)
        np.save('./test_labels/{}.npy'.format(ind), seg)

#prepare line
def prepare_line():
    train_data_names = read_data_names('./train_labels')
    test_data_names = read_data_names('./test_labels')

    train_images = read_images('./train_images', data_names=train_data_names)
    train_labels = read_labels('./train_labels')
    # train_data_names = read_data_names('./train_labels')

    test_labels = read_labels('./test_labels')
    test_images = read_images('./test_images', test_data_names)
    # test_data_names = read_data_names('./test_labels')

    for ind, image in enumerate(train_images):
        print(ind)
        H, W, C = image.shape
        lab = train_labels[ind]
        seg = draw_line(lab, H, W)
        np.save('./train_lines/{}.npy'.format(ind), seg)

    for ind, image in enumerate(test_images):
        print('test {}'.format(ind))
        H, W, C = image.shape
        lab = test_labels[ind]
        seg = draw_line(lab, H, W)
        np.save('./test_lines/{}.npy'.format(ind), seg)


    seg = np.load('./train_labels/1.npy')
    ln = np.load('./train_lines/1.npy')
    plt.figure()
    plot_image(seg, segmap = ln)
    plt.show()

def get_names():
    dirs = ['./train_labels', './test_labels']
    for d in dirs:
        npylist = [n for n in os.listdir(d) if n[-4:] == '.npy']
        l = len(npylist)
        from label_io import write_data_names
        dn = ['{}.npy'.format(i) for i in range(l)]
        write_data_names(dn, d, title = 'segmap_names')

    dirs = ['./train_lines', './test_lines']
    for d in dirs:
        npylist = [n for n in os.listdir(d) if n[-4:] == '.npy']
        l = len(npylist)
        from label_io import write_data_names
        dn = ['{}.npy'.format(i) for i in range(l)]
        write_data_names(dn, d, title = 'line_names')

if __name__ == '__main__':
    prepare_line()
    plt.figure()
    sample_line = np.load('./train_lines/1.npy')

    from label_io import hwc, chw
    plt.imshow(hwc(sample_line))
    plt.show()
