import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, ZeroPadding2D, BatchNormalization
from keras.utils import np_utils

 
#1 加载数据集，对数据集进行处理，把输入和结果分开
def load_data():

	train = pd.read_csv('train.csv')

	#把训练的图片（0～9）标签进行one-hot编码
	label = pd.get_dummies(train['label'])

	#丢弃label列，把训练的图片数据转化成28*28的图片，并进行归一化
	train_df = train.drop(['label'], axis=1)
	train_df = np.array(train_df)
	train_df = train_df.reshape(-1,28,28,1)
	train_df = train_df/255
	#df->DataFrame

	#把测试集的图片数据转化成28*28的图片，并进行归一化
	test = pd.read_csv('test.csv')
	test = np.array(test)
	test = test.reshape(-1,28,28,1)
	test = test/255

	return train_df, label, test

train, label, test = load_data()
 
#2 设相关参数
#设置对训练集的批次大小
batch_size = 64
#设置最大池化，池化核大小
pool_size = (2,2)
 
#3 定义网络，按照zeropadding,巻积层，规范层，池化层进行设置
cnn_net = Sequential()
 
cnn_net.add(Conv2D(32, kernel_size = (5,5),input_shape = (28,28,1), activation=('relu')))
cnn_net.add(BatchNormalization(epsilon = 1e-6, axis = 1))
cnn_net.add(MaxPool2D(pool_size = pool_size))
 
cnn_net.add(ZeroPadding2D((1,1)))
cnn_net.add(Conv2D(48, kernel_size = (3,3), activation=('relu')))
cnn_net.add(BatchNormalization(epsilon = 1e-6, axis = 1))
cnn_net.add(MaxPool2D(pool_size = pool_size))
 
cnn_net.add(Dropout(0.25))
cnn_net.add(Flatten())
 
cnn_net.add(Dense(1024, activation='relu'))
cnn_net.add(Dense(10,activation='softmax'))
 
#4 查看网络结构
cnn_net.summary()
 
#5 训练模型，保存模型
cnn_net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print('开始训练')
cnn_net.fit(train,label,batch_size=batch_size,epochs=15,verbose=1,validation_split=0.2)
print('训练结束')
cnn_net.save('cnn_net_model.h5')
 
#6 加载模型
cnn_net = load_model('cnn_net_model.h5')
 
#7 生成提交预测结果
prediction = cnn_net.predict_classes(test, batch_size=32, verbose=1)
np.savetxt('DeepConvNN.csv',np.c_[range(1,len(prediction)+1),prediction],delimiter=',',header='ImageId,Label',comments='',fmt='%d')