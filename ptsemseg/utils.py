"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np

from collections import OrderedDict

import matplotlib.pyplot as plt

# Note the requirement of imshow is (H,W,C)

#def show_images(images, cols = 1, titles = None):
#    """Display a list of images in a single figure with matplotlib.
#    
#    Parameters
#    ---------
#    images: List of np.arrays compatible with plt.imshow.
#    
#    cols (Default = 1): Number of columns in figure (number of rows is 
#                        set to np.ceil(n_images/float(cols))).
#    
#    titles: List of titles corresponding to each image. Must have
#            the same length as titles.
#    """
#    assert((titles is None)or (len(images) == len(titles)))
#    n_images = len(images)
#    print(n_images, "length of list")
#    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
#    fig = plt.figure()
#   
#    for n, (image, title) in enumerate(zip(images, titles)):
#    #    print(n, "the n")
#        #a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#        a = fig.add_subplot(np.ceil(n_images/float(cols)), cols , n + 1)    # bbox_inches='tight' here?
#    #    if image.ndim == 2:
#    #        plt.gray()
#    #    print(image.shape, "shape of image to imshow")
#        fig.tight_layout()
#
#        plt.imshow(image)
#        #a.set_title(title)
#        a.set_yticklabels([])
#        a.set_xticklabels([])
#    print(np.array(fig.get_size_inches()) * n_images, "here")
#    #fig.set_size_inches(np.array(fig.get_size_inches()) * n_images/7)
#    np_array = np.array(fig.get_size_inches())
#    np_array[0] = np_array[0] / 2
#    #fig.set_size_inches(np_array)
#    #plt.figure(figsize=(10,3))
#    plt.tight_layout()
#    plt.show() # bbox_inches='tight'
#    #fig.savefig('output.pdf', bbox_inches='tight')
#    #fig.savefig('output.png', bbox_inches='tight')


def show_images(images, save_path, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    #print(n_images, "length of list")
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure(figsize=(100,100))
    count = 1
    
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(1, cols, (n%3)+1)
        #a = fig.add_subplot(np.ceil(n_images/float(cols)), cols , n + 1)    # bbox_inches='tight' here?
    #    if image.ndim == 2:
    #        plt.gray()
    #    print(image.shape, "shape of image to imshow")
        #fig.tight_layout()
        plt.imshow(image)
        #a.set_title(title)
        a.set_yticklabels([])
        a.set_xticklabels([])
        if (n + 1) % 3 == 0:
            fig.savefig(save_path + "_ex{}.png".format(count), dpi='figure',bbox_inches='tight')
            count += 1
            plt.close()
            fig = plt.figure(figsize=(100,100))
    
    plt.close()
    



def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger
