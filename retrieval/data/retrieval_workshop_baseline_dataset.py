import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from util import custom_transforms
import torchvision.transforms as transforms
import pdb
import torch
import cv2

from util.smart_crop_transforms import RandomCropDramaticlly
from util import augment

class RetrievalWorkshopBaselineDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        opt.dataset_name = 'train_set.npy'
        opt.level3_dic = 'level3_dict.npy'
        opt.level2_dic = 'level2_dict.npy'

        if opt.phase != 'train':
            if opt.reverse:
                self.shapes_info = np.load(os.path.join(opt.dataroot, 'data_info', opt.dataset_name), allow_pickle=True)[::-1]
            else:
                self.shapes_info = np.load(os.path.join(opt.dataroot, 'data_info', opt.dataset_name), allow_pickle=True)
        else:
            if opt.reverse:
                self.shapes_info = np.load(os.path.join(opt.dataroot, 'data_info', opt.dataset_name), allow_pickle=True)[::-1]
            else:
                self.shapes_info = np.load(os.path.join(opt.dataroot, 'data_info', opt.dataset_name), allow_pickle=True)

        self.level3_dic = np.load(os.path.join(opt.dataroot, 'data_info', opt.level3_dic), allow_pickle=True).item()
        self.level2_dic = np.load(os.path.join(opt.dataroot, 'data_info', opt.level2_dic), allow_pickle=True).item() 
        
        self.level2_keys = list(self.level2_dic.keys())
        self.phase = opt.phase
        self.data_size = len(self.shapes_info)
        random.shuffle(self.shapes_info)
        
        self.unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.real_image_path = os.path.join(opt.dataroot, 'image_data')
        self.bicycle_image_path = os.path.join(opt.dataroot, 'notexture_pool_data')
        
        self.query_transform = transforms.Compose([
                transforms.RandomRotation(degrees=8, fill=234),
                transforms.Resize(320),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.pool_transform = transforms.Compose([
                transforms.RandomRotation(degrees=8, fill=234),
                transforms.Resize(320),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        opt.fine_size = 256
        self.fine_size = opt.fine_size
        self.phase = opt.phase
        self.query_random_erase = transforms.Compose([transforms.RandomErasing()])

    def __getitem__(self, index):
        """
        Get training samples
        
        query_img: random choosed rgb sample image;
        positive_img: rendered no-texture sample image, same as query_img;
        negative_img: rendered no-texture sample image, random choosed from leve3 or leve2 dict
        """
        shape_info = self.shapes_info[index % self.data_size]
        shape_id = shape_info['model']
        positive_id = shape_id
        image_name = shape_info['image']
        level2_id = shape_info['level2_label']
        level3_id = shape_info['level3_label']
        center_label = int(shape_info['model'])
        cate_label = level2_id
        
        query_img = self._load_query_image_notexture(shape_id, image_name)
 
        # choose negative id
        if random.random() > 0.15:
            if random.random() > 0.3:
                alter_pool = self.level3_dic[level3_id].copy()
            else:
                alter_pool = self.level2_dic[level2_id].copy()
        else:
            alter_pool = self.level2_dic[random.choice(self.level2_keys)].copy()
        
        if shape_id in alter_pool:
            alter_pool.remove(shape_id)

        if len(alter_pool) == 0:
            alter_pool = self.level2_dic[level2_id].copy()
            if shape_id in alter_pool:
                alter_pool.remove(shape_id)
        negative_id = random.choice(alter_pool)

        positive_img = self._load_pool_image_notexture(positive_id)
        negative_img = self._load_pool_image_notexture(negative_id)
        

        return {'query_img':query_img, 'positive_img':positive_img, 'negative_img':negative_img, 'center_label':center_label, 'cate_label':cate_label}

    def __len__(self):
        return self.data_size
    
    def _load_query_image(self, shape_id, image_name):
        
        prob = random.random()
        
        syn_id = '{0:03d}'.format(int(random.randint(0, 30)))
        if prob > 0.8:
            img_file = os.path.join(self.bicycle_image_path,  shape_id, 'image_' + syn_id + '.jpg')
        else:
            img_file = os.path.join(self.real_image_path, image_name)
        
        print('query: ', img_file)

        img = Image.open(img_file).convert('RGB')
        trans_img = self.query_transform(img)

        if random.random() > 0.5:
            return self.query_random_erase(trans_img)
        else:
            return trans_img

    def _load_query_image_notexture(self, shape_id, image_name):
        
        prob = random.random()
        
        syn_id = '{0:03d}'.format(int(0))
        if prob > 0.8:
            img_file = os.path.join(self.bicycle_image_path,  shape_id, 'image_' + syn_id + '.png')
        else:
            img_file = os.path.join(self.real_image_path, image_name)
        

        img = Image.open(img_file).convert('RGB')
        trans_img = self.query_transform(img)

        if random.random() > 0.5:
            return self.query_random_erase(trans_img)
        else:
            return trans_img


    def _load_pool_image_notexture(self, shape_id):
        syn_id = '{0:03d}'.format(int(0))
        prob = random.random()
    
        img_file = os.path.join(self.bicycle_image_path,  shape_id, 'image_' + syn_id + '.png')
        
        img = Image.open(img_file).convert('RGB')
        trans_img = self.pool_transform(img)

        return trans_img


    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
