import os
import json
import torch
import numpy as np
import scipy.misc as m

import sys

from torch.utils import data

from utils import recursive_glob

from PIL import Image

import scipy.misc
#from ptsemseg.augmentations import *

class facesLoader(data.Dataset):
    def __init__(self, root, split="training", img_size=(640, 1280), 
                 is_transform=True, augmentations=None):
        self.root = root     # root is path not tree, /home/brm512/datasets/vistas_v1.1/
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 12

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

        #print("got here")
        #print(self.class_colors)
        #if self.augmentations is not None:
        #    img, lbl = self.augmentations(img, lbl)

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
        #lbl[lbl == 65] = self.ignore_id
        
        #lbl[lbl>=4] = self.ignore_id
        return img, lbl
 

    def decode_segmap(self, temp):   # temp is HW np slice
        r = temp.copy() 
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.class_colors[l][0]
            g[temp == l] = self.class_colors[l][1]
            b[temp == l] = self.class_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))   # dummy tensor in np order HWC
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

if __name__ == '__main__':

    #np.set_printoptions(threshold=sys.maxsize)
    
    root = '/home/brm512/datasets/faces/labels'
    for looproot, _, filenames in os.walk(root, topdown=False):
        
        
        #print(looproot)
       # print(middle)
        #print(filenames)
        
        
        #example_path = os.path.join(root, '232194_1')
        #dst = facesLoader(local_path, img_size=(64, 64), is_transform=True, augmentations=None)
        #bs = 1
        #trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0, shuffle=False)
        
        #files = recursive_glob(rootdir=example_path, suffix='.png')
        
        #print(filenames)
        path = os.path.join(looproot, filenames[0])
        
        filenames.sort()
        w, h = Image.open(path).size
    
        #print(filenames, w, h)
        
        
        
        example_tensor = np.zeros((h ,w ,11))
        for index, file in enumerate(filenames):
            path = os.path.join(looproot, file)
            #print(path)
            
            class_slice = Image.open(path)
            #convert to np
            class_slice_np = np.array(class_slice)
            #print(class_slice_np.shape)
            #break
            
            example_tensor[:, :, index] = class_slice_np #assuming sort worked
        
            
        #print(example_tensor.shape)
        #print(example_tensor[170][220][:])
        
        
        
        
        
        #for pixel in example_tensor[]:
            
        #count = 0    
        #for i in range(h):
            #for j in range(w):
        final_classes = np.argmax(example_tensor, axis=2)
        
        
        #print(final_classes.shape)
        #print(final_classes[170][220])
        
        
        
        #scipy.misc.toimage(final_classes, cmin=0.0, cmax=...).save('/home/brm512/datasets/faces/processed/test.png')
        
        im = Image.fromarray(final_classes.astype('uint8'))
        save_path = os.path.join('/home/brm512/datasets/faces/processed', os.path.basename(looproot)) # joins the basename on
        save_path = save_path + '.png'
        #print(save_path)
        #break
        im.save(save_path)
        
        
    
#    count = 1
#    for looproot, _, filenames in os.walk(root):
#        print(looproot)
#        print(len(filenames))
#        print(filenames)
#        count = count + 1
#       # print(filenames)
        
        
        
        
        
        
   # print(count)
    
    #myImg = Image.open('/home/brm512/datasets/faces/processed/test.png')
    #example = np.array(myImg)
    #print(example[170][:])
    
    
    #print(indices[270][:])
    #print(type(indices))
            #print(example_tensor[i][j][:])
            #count = count +1
            
    #print(count)
    #print(h*w)
        #example_tensor[i,j,:]
    
    # check index
    
    #print(type(files))    
    
    #print(len(files))
    #print(files)
    
#    for file in files:
#        print(file)
    
    #for i, data in enumerate(trainloader):
    #    x = dst.decode_segmap(data[1][0].numpy())   # 1 isolates the second element of the tuple (label) and 0 isolates the first batch.
    #    print("batch :", i)
