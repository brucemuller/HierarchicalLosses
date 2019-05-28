import os
import json
import torch
import numpy as np
import scipy.misc as m

from torch.utils import data

from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *

class facesLoader(data.Dataset):
    def __init__(self, root, split="training", img_size=(640, 1280), 
                 is_transform=True, augmentations=None):
        self.root = root     
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 11

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        #self.mean = np.array([80.5423, 91.3162, 81.4312])
        self.files = {}
        
        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'labels')

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.jpg')
        self.number_of_images = len(self.files[split])

     #   self.class_ids, self.class_names, self.class_colors = self.parse_config()

        self.ignore_id = 250

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png")) # assume labels have same file name except use png

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl
        #return lbl
    
    def getLabel(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png"))
        lbl = Image.open(lbl_path)
        lbl = torch.from_numpy(np.array(lbl)).long()
        
        return lbl
        

    def transform(self, img, lbl):
        #if self.img_size == ('same', 'same'):
        #    pass
        #else: 
        #    img = img.resize((self.img_size[0], self.img_size[1]), 
        #                      resample=Image.LANCZOS)  # uint8 with RGB mode
        #    lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        
        img = img.resize((self.img_size[0], self.img_size[1]), 
                              resample=Image.LANCZOS)  # uint8 with RGB mode
        lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        
        img = np.array(img).astype(np.float64) / 255.0                                      # IS THIS NORMALISATION OK?
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # From HWC to CHW
        lbl = torch.from_numpy(np.array(lbl)).long()
        #lbl[lbl == 0] = self.ignore_id
        
        #lbl[lbl>=4] = self.ignore_id
        return img, lbl
 

    def decode_segmap(self, temp):   # temp is HW np slice
        Background = [0, 0, 0]
        Face_Skin = [128, 128, 128]    
        Left_Eyebrow = [128, 0, 0]   
        Right_Eyebrow = [192, 192, 128]
        Left_Eye = [255, 69, 0]
        Right_Eye = [128, 64, 128]
        Nose = [60, 40, 222]
        Upper_Lip = [128, 128, 0]
        Inner_Mouth = [192, 128, 128]
        Lower_Lip = [64, 64, 128]
        Hair = [64, 0, 128]

        label_colours = np.array(
            [
                Background,
                Face_Skin,
                Left_Eyebrow,
                Right_Eyebrow,
                Left_Eye,
                Right_Eye,
                Nose,
                Upper_Lip,
                Inner_Mouth,
                Lower_Lip,
                Hair
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
    
    def class_code_abstractor(self, temp_label, root, level = 0):     # May want to put this function somewhere else where the root already is
        
        mylist = [7,8,9,1,6,4,5,2,3]
        
        for index, pix_class in enumerate(temp_label):
            if pix_class in mylist:
                temp_label[index] = findParentClass(pix_class)
        
        return temp_label
            
        
    def decode_segmap_levelwise(self, temp):   # temp is HW np slice
        n_classes = 17
        
        Background = [0, 0, 0]
        Face_Skin = [128, 128, 128]    
        Left_Eyebrow = [128, 0, 0]   
        Right_Eyebrow = [192, 192, 128]
        Left_Eye = [255, 69, 0]
        Right_Eye = [128, 64, 128]
        Nose = [60, 40, 222]
        Upper_Lip = [128, 128, 0]
        Inner_Mouth = [192, 128, 128]
        Lower_Lip = [64, 64, 128]
        Hair = [64, 0, 128]
        
        Foreground = [255, 241, 20]
        Face = [60, 180, 75]
        Mouth = [255, 13, 241]
        Skin = [245, 130, 48]
        Eyes = [170, 255, 195]
        Eyebrows = [0, 128, 128]

        label_colours = np.array(
            [
                Background,
                Face_Skin,
                Left_Eyebrow,
                Right_Eyebrow,
                Left_Eye,
                Right_Eye,
                Nose,
                Upper_Lip,
                Inner_Mouth,
                Lower_Lip,
                Hair,
                Foreground,
                Face,
                Mouth,
                Skin,
                Eyes,
                Eyebrows
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        
        for l in range(0, n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb
