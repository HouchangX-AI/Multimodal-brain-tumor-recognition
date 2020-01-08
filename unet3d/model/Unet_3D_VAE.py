# import 
import sys 
import os 
sys.path.append(os.getcwd())
import tensorflow as tf 
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Activation, BatchNormalization, PReLU, concatenate
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import Model, Input 
from tensorflow.keras import backend as K

from unet3d.metrics import dice_coefficient, dice_coefficient_loss, weighted_dice_coefficient, weighted_dice_coefficient_loss
from config import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cfg = tf.ConfigProto()
# cfg.gpu_options.per_process_gpu_memory_fraction = 0.6 # 占用GPU90%的显存
# session = tf.Session(config=cfg)

# 设置channels_first --->（c, x, y, z)
K.set_image_data_format("channels_first")

# FIXME try PReLu
def Unet3D_Model(inputs_shape):
    # 定义输入
    input_layer = Input(inputs_shape)
    # Conv3D-->BatchNormalization-->Activation * 2
    x1 = Conv3D(filters=32, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    # M + CBA * 2 (缩小特征图尺度)
    x2 = MaxPooling3D(pool_size=(2,2,2))(x1)
    x2 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    # M + CBA * 2 (缩小特征图尺度)
    x3 = MaxPooling3D(pool_size=(2,2,2))(x2)
    x3 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    # M + CBA * 2 (缩小特征图尺度)
    x4 = MaxPooling3D(pool_size=(2,2,2))(x3)
    x4 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv3D(filters=512, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    # CT + ca + CBA * 2 (增大特征图尺度+跳跃式链接)
    x5 = Conv3DTranspose(filters=512, kernel_size=(2,2,2), padding='same', strides=(2,2,2))(x4)
    x5 = concatenate([x5, x3], axis=1)
    x5 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x5)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = Conv3D(filters=256, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x5)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    # CT + ca + CBA * 2 (增大特征图尺度+跳跃式链接)
    x6 = Conv3DTranspose(filters=256, kernel_size=(2,2,2), padding='same', strides=(2,2,2))(x5)
    x6 = concatenate([x6, x2], axis=1)
    x6 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x6)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv3D(filters=128, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x6)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    # CT + ca + CBA * 2 (增大特征图尺度+跳跃式链接)
    x7 = Conv3DTranspose(filters=128, kernel_size=(2,2,2), padding='same', strides=(2,2,2))(x6)
    x7 = concatenate([x7, x1], axis=1)
    x7 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x7)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    x7 = Conv3D(filters=64, kernel_size=(3,3,3), padding='same', strides=(1,1,1))(x7)
    x7 = BatchNormalization()(x7)
    x7 = Activation('relu')(x7)
    #
    x8 = Conv3D(filters=1, kernel_size=(1,1,1), padding='same', strides=(1,1,1))(x7)
    x8 = Activation('sigmoid')(x8)

    model = Model(inputs=input_layer, outputs=x8)
    model.compile(optimizer=Adam(lr=config['initial_learning_rate']), loss=weighted_dice_coefficient_loss, metrics=[weighted_dice_coefficient])
    return model

if __name__ == "__main__":
    inputs_shape = (4,144,144,144)
    model = Unet3D_Model(inputs_shape)
    model.summary()



    