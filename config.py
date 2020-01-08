import sys 
import os 
sys.path.append(os.getcwd())


config = dict()

# 模态
config['modalities'] = ['t1', 't1ce', 'flair', 't2'] # data模态标签
config['label_modalities'] = ['truth'] # label标签
config['all_modalities'] = config['modalities'] + config['label_modalities'] # 总标签

# 路径
config['preprocess_1_dir'] = 'data/preprocessed_1' # 一次处理保存路径
config['preprocess_2_dir'] = 'data/preprocessed_2/brats_data.h5' # 二次处理保存路径
#config["data_file"] = os.path.abspath('brats_data.h5') # 

# 输入图像信息
config['new_shape'] = (144,144,144) # 输入图像形状
config['input_channels'] = len(config['modalities']) # 输入图像通道数
config["labels"] = (1, 2, 4) # 标签内容
config["input_shape"] = tuple([config["modalities"]] + list(config["new_shape"]))

# 训练参数
config['data_format'] = 'channels_first' # 通道在前

# 模型超参
config["epochs"] = 500 # 训练次数
config["batch_size"] = 1 # batch
config['initial_learning_rate'] = 0.00001 # 初始化学习率
config["model_file"] = os.path.abspath("isensee_2017_model.h5")











config['input_shape'] = tuple([config["input_channels"]] + list(config["new_shape"]))
config["validation_batch_size"] = 1


config["patience"] = 10 
config["early_stop"] = 50 

config["learning_rate_drop"] = 0.5 
config["train_split"] = 0.8

