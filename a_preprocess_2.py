# import
import sys 
import os 
sys.path.append(os.getcwd()) 
import nibabel as nib 
import numpy as np 
import tables 
import glob 

from nilearn.image.image import _crop_img_to as crop_img_to
from a_preprocess.Remove_black_frame import remove_black_frame
from a_preprocess.normalize import normalize_data
from config import config 
from a_preprocess.resize import resize


def get_nii_path():
    all_niis_path = list()
    folders_name = list()
    for folder_name in glob.glob(os.path.join(os.path.dirname(__file__), "data", "preprocessed_1", "*", "*")):
        folders_name.append(os.path.basename(folder_name))
        nii_path = list()
        for modality in config['all_modalities']:
            # nii_path.append(os.path.join(folder_name, os.path.basename(folder_name) + '_' + modality + ".nii.gz"))
            nii_path.append(os.path.join(folder_name, modality + ".nii.gz"))
        all_niis_path.append(tuple(nii_path))
    return all_niis_path, folders_name

def create_data_file(out_file, channels, samples, image_shape):
    hdf5_file = tables.open_file(out_file, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_shape = tuple([0, channels] + list(image_shape))
    label_shape = tuple([0, 1] + list(image_shape))
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float16Atom(), shape=data_shape, filters=filters, expectedrows=samples)
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=label_shape, filters=filters, expectedrows=samples)
    affine_storage = hdf5_file.create_earray(hdf5_file.root, 'affine', tables.Float16Atom(), shape=(0, 4, 4), filters=filters, expectedrows=samples)
    return hdf5_file, data_storage, label_storage, affine_storage

def add_data_to_storage(data_storage, truth_storage, affine_storage,image,label,affine):
    data_storage.append(np.asarray(image)[np.newaxis])
    truth_storage.append(np.asarray(label, dtype=np.uint8)[np.newaxis][np.newaxis])
    affine_storage.append(np.asarray(affine)[np.newaxis])

def  image_preproccess(all_niis_path,data_storage, label_storage, affine_storage):
    # 路径循环
    N = len(all_niis_path)
    n = 1
    for five_nii_path in all_niis_path:
        
        # 找label的位置
        index = 0
        for i, one_input in enumerate(five_nii_path):
            if 'truth.nii.gz' in one_input:
                index = i
                # print(one_input)
        # 切黑框
        without_black_frame_images = remove_black_frame(five_nii_path)
        # resize
        resized_images = list()
        for i, img in enumerate(without_black_frame_images):
            if i == index:
                new_shape_img = resize(img, config['new_shape'], interpolation='linear')
            else:
                new_shape_img = resize(img, config['new_shape'], interpolation='nearest')
            resized_images.append(new_shape_img)
        # normalize
        label = np.array(resized_images[index].get_fdata())
        #
        del resized_images[index]
        normalize_image = normalize_data(resized_images)
        affine = nib.load(five_nii_path[0]).affine
        #
        add_data_to_storage(data_storage, label_storage, affine_storage,normalize_image,label,affine)
        print('处理了：',round(n/N,2)*100,'%')
        n += 1

def main():
    all_niis_path, folders_name = get_nii_path()

    out_file = config['preprocess_2_dir']
    image_shape = config['new_shape']
    samples = len(all_niis_path)
    channels = config['input_channels']
    hdf5_file, data_storage, label_storage, affine_storage = create_data_file(out_file, channels, samples, image_shape)
    #
    image_preproccess(all_niis_path, data_storage, label_storage, affine_storage)
    #
    hdf5_file.create_array(hdf5_file.root, 'folders_name', obj=folders_name)
    hdf5_file.close()


if __name__ == "__main__":
    main()















