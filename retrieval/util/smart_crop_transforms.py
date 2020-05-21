"""
Crop image randomly based on original size
"""
import os
import cv2
import numpy as np
import random
from torchvision import transforms


class RandomCropDramaticlly(object):
    def __init__(self, minimum_percent=0.7, maximum_percent=0.85, p=0.5):
        """
        random crop input image by percent, returned image have same size with input image

        the cropped image is between minimum percent and maximum percent
        """
        self.minimum_percent = minimum_percent
        self.maximum_percent = maximum_percent
        self.w_orig = None
        self.h_orig = None
        self.bbox_img = None    # is image in bbox type
        self.p = p

    def __call__(self, input_img):
        """
        crop image randomly
        """
        if random.random() < self.p: return input_img

        self.h_orig = input_img.shape[0]
        self.w_orig = input_img.shape[1]

        img = self.find_bbox_img(input_img)
        h, w = img.shape[0], img.shape[1]

        # crop image randomly by percent
        crop_img = self.random_crop_by_percent(img, w, h)

        # resize image
        if self.w_orig >= self.h_orig:
            resize = self.w_orig - 2 * 10
        else:
            resize = self.h_orig - 2 * 10
        crop_img = self.resize_by_scale(crop_img, resize, interpolation=None)

        # padding
        pad_img = self.padding(crop_img, input_img)

        # # visual
        # show_img = pad_img[:, :, 0:3]
        # show_sketch = pad_img[:, :, 3:]
        # cv2.imshow('process', show_img)
        # cv2.imshow('show_sketch', show_sketch)
        # cv2.waitKey(0)

        return pad_img

    def is_bbox_img(self, img):
        if (img[0][0][:] == [255, 255, 255]).all():
            return True
        else:
            return False

    def find_bbox_img(self, img):

        def find_bbox(img):
            if img.shape[2] == 3:
                indexs = np.where(img != [255, 255, 255])
            elif img.shape[2] == 1:
                indexs = np.where(img != 255)
            else:       # multi-channel
                rgb_img = img[:, :, 0:3]
                indexs = np.where(rgb_img != 255)
            min_x, max_x = min(indexs[0]), max(indexs[0])
            min_y, max_y = min(indexs[1]), max(indexs[1])
            return min_x, max_x, min_y, max_y
        min_x, max_x, min_y, max_y = find_bbox(img)
        bbox_img = img[min_x:max_x, min_y:max_y]
        return bbox_img

    def random_crop_by_percent(self, bbox_img, w, h):
        ratio_wh = w / h
        # extrem ratio condition
        if ratio_wh >= 5:
            cropped_size = int(w * self.minimum_percent)
            b_vertical = True
        elif ratio_wh <= 0.2:
            cropped_size = int(h * self.minimum_percent)
            b_vertical = False
        else:
            random_percent = random.randint(int(self.minimum_percent * 100), int(self.maximum_percent * 100)) * 0.01
            if random.random() < 0.5:
                cropped_size = int(w * random_percent)
                b_vertical = True
            else:
                cropped_size = int(h * random_percent)
                b_vertical = False

        if b_vertical:  # crop
            if random.random() < 0.5:
                crop_img = bbox_img[:, 0:cropped_size]
            else:
                crop_img = bbox_img[:, w - cropped_size:w]
        else:
            if random.random() < 0.5:
                crop_img = bbox_img[0:cropped_size, :]
            else:
                crop_img = bbox_img[h - cropped_size: h, :]
        return crop_img

    def resize_by_scale(self, img, size, interpolation=None):
        """
        interpolation: INTER_NEAREST, INTER_LINEAR, INTER_AREA
        """
        h, w = img.shape[0], img.shape[1]
        if (w >= h and w == size) or (h >= w and h == size):
            return img
        if w > h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return cv2.resize(img, (ow, oh), interpolation)

    def padding(self, crop_img, input_img):
        # padding
        h_diff = int((self.h_orig - crop_img.shape[0]) / 2)
        w_diff = int((self.w_orig - crop_img.shape[1]) / 2)
        if h_diff < 0: h_diff = 10
        if w_diff < 0: h_diff = 10

        replicate = crop_img
        for i in range(crop_img.shape[2]):
            padding_img = cv2.copyMakeBorder(crop_img[:, :, i], h_diff, h_diff, w_diff, w_diff, cv2.BORDER_CONSTANT,
                                             value=[255])
            padding_img = np.expand_dims(padding_img, axis=2)
            if i == 0:
                replicate = padding_img
            else:
                replicate = np.concatenate((replicate, padding_img), axis=2)

        if replicate.shape != input_img.shape: #img.shape:
            replicate = cv2.resize(replicate, (self.w_orig, self.h_orig))
        return replicate

def __repr__(self):
    return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


if __name__ == '__main__':
    img_dir = '/Users/shunming/data/3D/maijiaxiu/online/train/images'
    # img_dir = '/Users/shunming/data/3D/xilin_online_test/xiling_images'
    img_files = os.listdir(img_dir)
    random_crop = RandomCropDramaticlly()
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_dir, img_file))
        norm = cv2.imread(os.path.join(img_dir, img_file))
        # img = cv2.imread('/Users/shunming/data/3D/render_shape_images/new_process/render/3c0c52b6-84c8-465b-9d86-9f195c99f6d9/_r_000_albedo.png0001.png')
        # norm = cv2.imread('/Users/shunming/data/3D/render_shape_images/new_process/render/3c0c52b6-84c8-465b-9d86-9f195c99f6d9/_r_000_normal.png0001.png')
        img = cv2.resize(img, (448, 448))
        norm = cv2.resize(norm, (448, 448))
        img = np.concatenate((img, norm), axis=2)
        t = transforms.Compose([
            RandomCropDramaticlly()
        ])
        t(img)

