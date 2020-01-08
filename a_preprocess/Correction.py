import sys 
import os 
sys.path.append(os.getcwd())
from nipype.interfaces.ants import N4BiasFieldCorrection
import SimpleITK as sitk
import numpy as np

def Do_N4BiasFieldCorrection(in_path, out_path, image_type=sitk.sitkFloat64):
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_path
    correct.inputs.output_image = out_path
    # 
    try:
        done = correct.run()
        return done.outputs.output_image
    except:
        input_image = sitk.ReadImage(in_path, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_path)