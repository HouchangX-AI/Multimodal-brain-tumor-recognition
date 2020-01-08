
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

def get_image_foreground(files, background=0):
    foreground = np.zeros_like(nib.load(files[0]).get_fdata(),dtype=np.uint8)
    for image_file in files:
        image = nib.load(image_file)
        is_foreground = np.logical_not(image.get_data() == background)
        foreground[is_foreground] = 1
        return new_img_like(image, foreground)

def get_index_of_slice(foreground):
    img = check_niimg(foreground)
    image = img.get_data()
    #
    all_index = np.array(np.where(image == 1))
    #
    min_index = all_index.min(axis=1)
    max_index = all_index.max(axis=1) + 1
    #
    min_index = np.maximum(min_index - 1, 0)
    max_index = np.minimum(max_index + 1, image.shape[:3])
    #
    slices = [slice(s, e) for s, e in zip(min_index, max_index)]
    return slices

def remove_black_frame(files):
    foreground = get_image_foreground(files)
    crop = get_index_of_slice(foreground)
    new_image_files = list()
    for image_file in files:
        image = nib.load(image_file)
        image_croped = crop_img_to(image,crop,copy=True)
        new_image_files.append(image_croped)
    return new_image_files


if __name__ == "__main__":

    set_of_files = ['f_data/original/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz','f_data/original/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t2.nii.gz']
    nii_foreground=get_image_foreground(set_of_files)
    crop = get_index_of_slice(nii_foreground)
    # print(crop)
    nii_image = nib.load('f_data/original/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz')
    nii_image_croped=crop_img_to(nii_image,crop,copy=True)

    nib.save(nii_image_croped,'simpleitk_save5.nii.gz')



