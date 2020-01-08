import sys 
import os 
sys.path.append(os.getcwd())
from nilearn.image.image import _crop_img_to as crop_img_to
from nilearn.image.image import check_niimg
from nilearn.image import new_img_like
import matplotlib.pyplot as plt
import SimpleITK as sitk 
import nibabel as nib
import numpy as np 

def normalize_data(files, background=0):
    means = list()
    stds = list()
    for image in files:
        image = image.get_fdata()
        # img = np.reshape(image,(image.shape[0]*image.shape[1]*image.shape[2]))
        index = image[image > 0]
        mean = index.mean()
        std  = index.std()
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    mean = np.mean(means)
    std = np.mean(stds)
    # print('mean: ',mean,'   std: ',std)
    new_files = list()
    for image in files:
        image = image.get_fdata()
        out_img = (image - mean)/std
        zeros_volume = np.zeros(out_img.shape)
        out_img[image == 0] = zeros_volume[image == 0]
        new_files.append(out_img)
    new_files = np.array(new_files)
    return new_files

