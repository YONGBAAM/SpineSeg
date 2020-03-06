import torchvision.transforms as tr
import torchvision.transforms.functional as TF
import sys
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import collections

from label_io import to_relative, to_absolute

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

    _pil_interpolation_to_str = {
        Image.NEAREST: 'PIL.Image.NEAREST',
        Image.BILINEAR: 'PIL.Image.BILINEAR',
        Image.BICUBIC: 'PIL.Image.BICUBIC',
        Image.LANCZOS: 'PIL.Image.LANCZOS',
        Image.HAMMING: 'PIL.Image.HAMMING',
        Image.BOX: 'PIL.Image.BOX',
    }




##################################
#
#   Refactoring Finished
#
#   모두 PIL 이미지 기준으로 한 트랜스폼임.
#
#   Label 다룰때는 맨처음 라벨 카피 해주기!
#
#################################

class SegResize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        r_image = TF.resize(image, self.size, self.interpolation)
        r_label = TF.resize(label, self.size, self.interpolation)


        return r_image, r_label

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)




class CoordRandomRotate:
    def __init__(self,max_angle, expand = False, is_random = True):
        self.max_angle = max_angle
        self.expand = expand
        self.is_random = is_random
    def __call__(self, image, label):
        if self.is_random:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
        else:
            angle = self.max_angle

        W1, H1 = image.size

        if self.expand:
            image = image.rotate(angle, expand=True)
            W2, H2 = image.size
            label = rotate_label(label, angle, H = H1, W = W1, new_centerXY=(W2 / 2, H2 / 2))

        else:
            r_label = rotate_label(label, angle, H = H1, W = W1)
            if not is_outlier(label, H1, W1, margin = 1):
                label = r_label
                image = image.rotate(angle, expand=False)

        return image, label

class CoordVerticalFlip:
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, label):
        W,H = image.size

        if np.random.random() < self.prob:
            r_img = TF.vflip(image)
            label = label.reshape(-1, 2)
            r_label = np.zeros_like(label)
            r_label[:, 0] = label[:, 0]
            r_label[:, 1] = H - 1 - label[:, 1]
            r_label = r_label.reshape(-1)
            return r_img, r_label
        else:
            return image, label

class CoordHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        W, H = image.size

        if np.random.random() < self.prob:
            r_img = TF.hflip(image)
            label = label.reshape(-1, 2)
            r_label = np.zeros_like(label)
            r_label[:,1] = label[:,1]
            r_label[:, 0] = W - 1 - label[:, 0]
            r_label = r_label.reshape(-1)
            return r_img, r_label
        else:
            return image, label

class CoordResize:
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, label):
        """
        Args:
            image (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        W_ori, H_ori = image.size
        H_target, W_target = self.size[0], self.size[1]
        r_image = TF.resize(image, self.size, self.interpolation)
        label = label.reshape(-1,2)
        r_label = np.zeros_like(label)
        r_label[:, 0] = label[:,0]*W_target / W_ori
        r_label[:, 1] = label[:,1]*H_target / H_ori
        r_label = r_label.reshape(-1)
        return r_image, r_label

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class CoordLabelNormalize:
    def __init__(self):
        pass

    def __call__(self, image, label):
        W,H = image.size
        r_label = to_relative(label, H=H, W=W)
        return image, r_label

class CoordJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.jitter = tr.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, label):
        r_image = self.jitter(image)
        return r_image, label

class SegCustomPad:
    def __init__(self, HoverW, fixH=True):
        self.ratio = HoverW
        self.fixH = fixH
        self.fixW = not fixH

    def __call__(self, image, segmap):
        W, H = image.size

        desire_W = int(H / self.ratio)
        left_pad = int((desire_W - W) / 2)
        right_pad = desire_W - W - left_pad
        im_pad = TF.pad(image, padding=(left_pad, 0, right_pad, 0), padding_mode='constant')
        seg_pad = TF.pad(image, padding=(left_pad, 0, right_pad, 0), padding_mode='constant')
        return im_pad, seg_pad

def rotate_label(label, degree, H, W, new_centerXY=None):
    theta = degree / 180 * np.pi
    s = np.sin(-theta)  # x y 축 바뀜
    c = np.cos(-theta)
    rot_matrix = [[c, s], [-s, c]]  # 회전행렬 transpose
    r_label = label.reshape(-1, 2)
    origin = np.asarray([W / 2, H / 2])  # x y좌표
    r_label = r_label - origin
    r_label = np.dot(r_label, rot_matrix)
    if new_centerXY is None:
        new_centerXY = origin
    r_label += new_centerXY
    r_label = r_label.reshape(-1)
    return r_label

def is_outlier(label, H, W, margin=0):
    label_r = label.reshape(-1,2)

    L = label_r[:,0] <0 + margin
    R = label_r[:,0] >W -margin
    U = label_r[:,1] <0 +margin
    D = label_r[:,1] >H -margin
    crit = np.sum(L) + np.sum(R) + np.sum(U) + np.sum(D)
    if crit ==0:
        return False
    else:
        return True

#if __name__ == '__main__':
    # #def label-to-image
    # data_path = './highres_images'
    # label_path = './highres_labels'
    #
    # labels = read_labels(location = label_path)
    # data_names = read_data_names(location = label_path)
    #
    # batch_size = 8
    #
    # customTransforms = [
    #         CoordJitter(brightness=0, contrast=0, saturation=0, hue=0)
    #         CoordCustomPad(512 / 256),
    #         CoordResize((512, 256)),
    #         CoordLabelNormalize()
    #     ]
    #
    #     dset = CoordDataset(data_path, labels, data_names, transform_list=customTransforms)
    #     loader_transform = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    #     index = 0
    #     for val_data in loader_original:
    #         img = val_data['image'].cpu().to(dtype=torch.float)[0]  # for batch size 1
    #         lab = val_data['label'].cpu().to(dtype=torch.float)[0]
    #         img = img.numpy()
    #         lab = lab.numpy()
    #         C, H, W = img.shape
    #
    #         if is_outlier(lab, H=H, W=W, margin=1):
    #             if index not in outliers_count.keys():
    #                 outliers_count[index] = []
    #             outliers_count[index].append((img, lab, angle))
    #         index += 1
    #
    # transform = tr.Compose([
    #     tr.RandomRotation(30, expand = False)
    #     tr.ToTensor()
    # ])


    # customTransforms = [
    #                     CoordRandomRotate(max_angle = 5, expand = True, is_random =False),
    #                     # CoordHorizontalFlip(0.5),
    #                     # CoordVerticalFlip(0.5),
    #                     CoordCustomPad(512 / 256),
    #                     CoordResize((512, 256)),
    #                     CoordLabelNormalize()
    #                     ]
    #
    # data_path = './resized_images'
    # label_path = './resized_labels'
    # for ind, data_name in enumerate(data_names):
    #     img = Image.open(os.path.join(data_path, data_names[ind]))
    #     seg = np.load(os.path.join(label_path, data_name + '.npy'))
    #     seg = tr.ToPILImage(seg)
    #     img, seg = customTransforms[0](img, seg)
    #     plt.figure()
    #     plt.subplot(211)
    #     plt.imshow(img)
    #     plt.subplot(212)
    #     plt.imshow(seg)
    #     plt.show()


    # import numpy as np
    # max_angle = 8
    #
    # angles = np.arange(-max_angle, max_angle+0.1, 0.5)
    #
    # outliers_count = {}
    # for angle in angles:
    #     print('processing angle {}'.format(angle))
    #     customTransforms = [
    #         CoordRandomRotate(angle, expand=False, is_random=False),
    #         CoordCustomPad(512 / 256),
    #         CoordResize((512, 256)),
    #         CoordLabelNormalize()
    #     ]
    #
    #     dset = CoordDataset(data_path, labels, data_names, transform_list=customTransforms)
    #     loader_original = DataLoader(dataset=dset, batch_size=1, shuffle=False)
    #     index = 0
    #     for val_data in loader_original:
    #         img = val_data['image'].cpu().to(dtype=torch.float)[0]  # for batch size 1
    #         lab = val_data['label'].cpu().to(dtype=torch.float)[0]
    #         img = img.numpy()
    #         lab = lab.numpy()
    #         C, H, W = img.shape
    #
    #         if is_outlier(lab, H=H, W=W, margin=1):
    #             if index not in outliers_count.keys():
    #                 outliers_count[index] = []
    #             outliers_count[index].append((img, lab, angle))
    #         index += 1
    #
    # for ind, out_list in outliers_count.items():
    #     print('img{}, outliers {}'.format(ind, len(out_list)))
    #     for img, lab, angle in out_list:
    #         plt.figure()
    #         title = 'im_{}_angle_{}'.format(ind, angle)
    #         plt.title(title)
    #         plot_image(img, coord=lab)
    #         plt.savefig(os.path.join('./plots', title + '.png'))
    #         plt.close()
