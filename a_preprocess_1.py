import sys 
import os 
sys.path.append(os.getcwd()) 
import numpy as np 
import shutil 

from config import config 
from a_preprocess.Correction import Do_N4BiasFieldCorrection

def Image_registration(input_folder, output_folter, No_registration='flair', overwrite=False, find_label=True):

    All_Pathological_Type = ['HGG', 'LGG']
    # 寻找图片文件夹路径
    for Pathological_type in All_Pathological_Type:
        for img_folder in os.listdir(os.path.join(input_folder, Pathological_type)):
            input_img_folder_path = os.path.join(input_folder, Pathological_type, img_folder)
            output_img_folder_path = os.path.join(output_folter, Pathological_type, img_folder)
            # 是否覆盖
            if not os.path.exists(output_img_folder_path) or overwrite:
                if not os.path.exists(output_img_folder_path):
                    os.makedirs(output_img_folder_path)
                # 寻找图片路径
                for suffix in config["data_file_suffix"]:
                    input_img_path = os.path.join(input_img_folder_path, "*" + suffix + ".nii.gz")
                    output_img_path = os.path.join(output_img_folder_path, "*" + suffix + ".nii.gz")
                    if suffix != No_registration:
                        print('N4')
                        # Do_N4BiasFieldCorrection(input_img_path, output_img_path, image_type=sitk.sitkFloat64)
                    else:
                        print('copy')
                        # shutil.copy(input_img_path, output_img_path)
                if find_label:
                    input_img_path = os.path.join(input_img_folder_path, "*seg" + ".nii.gz")
                    output_img_path = os.path.join(output_img_folder_path, "*seg" + ".nii.gz")
                    # shutil.copy(input_img_path, output_img_path)
                    print('label')
                    pass

if __name__ == '__main__':
    Image_registration('f_data/original', 'f_data/preprocessed_N4B', overwrite=True, find_label=True) 
