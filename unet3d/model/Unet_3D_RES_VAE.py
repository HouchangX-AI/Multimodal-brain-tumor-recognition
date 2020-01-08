# import 
import sys 
import os 
sys.path.append(os.getcwd())
import tensorflow as tf 
from tensorflow.keras.layers import Conv3D, Flatten, Conv3DTranspose, UpSampling3D, Activation, BatchNormalization, PReLU, Dense, add, Lambda, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K

# from config import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cfg = tf.ConfigProto()
# cfg.gpu_options.per_process_gpu_memory_fraction = 0.6 # 占用GPU90%的显存
# session = tf.Session(config=cfg)

# 设置channels_first --->（c, x, y, z)
K.set_image_data_format("channels_first")

def res_block(input, filters=32, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1)):

    x1 = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(input)
    x_shortcut = Conv3D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(x1)
    x_shortcut = BatchNormalization()(x_shortcut)
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', strides=strides)(x2)
    x2 = BatchNormalization()(x2)
    #
    x4 = add([x2, x_shortcut])
    x4 = Activation('relu')(x4)
    return x4

def sampling(mean_log_var):
    mean, log_var= mean_log_var
    epsilon = K.random_normal(shape=(K.shape(mean)))
    return mean + K.exp(0.5 * log_var) * epsilon

def encoder_decoder_part(x):
    # inputs_shape: 1*4*160*192*128
    #
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*32*160*192*128
    x = res_block(x, filters=32) # 1*32*160*192*128
    #
    x1 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x) # 1*64*80*96*64
    x1 = res_block(x1, filters=64) # 1*64*80*96*64
    x1 = res_block(x1, filters=64) # 1*64*80*96*64
    #
    x2 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x1) # 1*128*40*48*32
    x2 = res_block(x2, filters=128) # 1*128*40*48*32
    x2 = res_block(x2, filters=128) # 1*128*40*48*32
    #
    x3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x2) # 1*256*20*24*16
    x3 = res_block(x3, filters=256) # 1*256*20*24*16
    x3 = res_block(x3, filters=256) # 1*256*20*24*16
    x3 = res_block(x3, filters=256) # 1*256*20*24*16
    x3 = res_block(x3, filters=256) # 1*256*20*24*16
    #
    x4 = Conv3D(filters=128, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x3) # 1*128*20*24*16
    x4 = UpSampling3D(size=(2, 2, 2))(x4) # 1*128*40*48*32
    x5 = add([x4, x2]) # 1*128*40*48*32
    x5 = res_block(x5, filters=128) # 1*128*40*48*32
    #
    x5 = Conv3D(filters=64, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x5) # 1*64*40*48*32
    x5 = UpSampling3D(size=(2, 2, 2))(x5) # 1*64*80*96*64
    x6 = add([x5, x1]) # 1*64*80*96*64
    x6 = res_block(x6, filters=64) # 1*64*80*96*64
    #
    x6 = Conv3D(filters=32, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x6) # 1*32*80*96*64
    x6 = UpSampling3D(size=(2, 2, 2))(x6) # 1*32*160*192*128
    x7 = add([x6, x]) # 1*32*160*192*128
    x7 = res_block(x7, filters=32) # 1*32*160*192*128
    #
    x7 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same')(x7) # 1*3*160*192*128
    x7 = Activation('sigmoid')(x7) # 1*3*160*192*128

    return x3, x7

def vae_part(x, dim=128):
    # inputshape: 1*256*20*24*16
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(x) # 1*16*10*12*8
    x = BatchNormalization()(x) # 1*16*10*12*8
    x = Activation('relu')(x) # 1*16*10*12*8
    #
    x = Flatten()(x) # 1*15360*1?
    x = Dense(256)(x) # 1*256*1?
    z_mean = Dense(dim)(x) # 1*128*1?
    z_log_var = Dense(dim)(x) # 1*128*1?
    x = Lambda(sampling, output_shape=(dim,))([z_mean, z_log_var]) # 128
    #
    x = Dense(15360)(x) # 15360
    x = Reshape((16, 10, 12, 8))(x) # 1*16*10*12*8
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*128*10*12*8
    #
    x = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*256*10*12*8
    x = Activation('relu')(x) # 1*256*10*12*8
    x = UpSampling3D(size=(2, 2, 2))(x) # 1*256*20*24*16
    #
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*128*20*24*16
    x = UpSampling3D(size=(2, 2, 2))(x) # 1*128*40*48*32
    #
    x = res_block(x, filters=128) # 1*128*40*48*32
    #
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*64*40*48*32
    x = UpSampling3D(size=(2, 2, 2))(x) # 1*64*80*96*64
    #
    x = res_block(x, filters=64) # 1*64*80*96*64
    #
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*32*80*96*64
    x = UpSampling3D(size=(2, 2, 2))(x) # 1*32*160*192*128
    #
    x = Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x) # 1*4*160*192*128
    return x, z_mean, z_log_var

def Unet_3D_RES_VAE(inputs_shape):
    input_layer = Input(inputs_shape)
    x3, x7 = encoder_decoder_part(input_layer)
    y_vae, z_mean, z_log_var = vae_part(x3)

    model = Model(inputs=input_layer, outputs=(x7, y_vae))
    return model

def model_loss(x_true, y_true, y_pred, vae_pred, z_mean, z_log_var):

    dice_loss = 1 - K.mean(2. * (K.sum(y_true * y_pred, axis=(-3, -2, -1)) + 1e-5 /2)/(K.sum(y_true, axis=(-3, -2, -1)) + K.sum(y_pred, axis=(-3, -2, -1)) + 1e-5))
    l2_loss = mean_squared_error(x_true, vae_pred)
    KL_loss = K.mean(tf.matmul(z_mean, z_mean)) + K.mean(tf.matmul(z_log_var, z_log_var)) + K.mean(K.log(tf.matmul(z_log_var, z_log_var))) - 1
    LOSS = dice_loss + 0.1 * l2_loss + 0.1 * KL_loss
    return LOSS



if __name__ == "__main__":
    inputs_shape = (4,160,192,128)
    model = Unet_3D_RES_VAE(inputs_shape)
    model.summary()

from keras.losses import mean_squared_error



















