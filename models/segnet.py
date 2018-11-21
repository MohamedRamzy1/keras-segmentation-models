# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 00:21:40 2018

@author: Mohamed
"""

from keras.models import Model, load_model
from keras.layers import *

class SegNet:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _downsample(self , inputs):
        avg_pool = AveragePooling2D(pool_size = (2 , 2) , strides = 2 , padding = 'same')(inputs)
        return avg_pool
    
    def _convlayer(self, inputs , filters , kernel_size , strides , activation = 'relu'):
        if(kernel_size == 3):
            X = Conv2D(filters , kernel_size = kernel_size , strides = strides , padding = 'same')(inputs)
        else:
            X = Conv2D(filters , kernel_size = kernel_size , strides = strides , padding = 'valid')(inputs)
        X = BatchNormalization()(X)
        X = Activation(activation)(X)
        if(activation != 'sigmoid'):
            sq = GlobalAveragePooling2D()(X)
            sq = Dense(int(filters / 4) , activation = 'relu')(sq)
            sq = Dense(filters , activation = 'sigmoid')(sq)
            sq = Reshape((1,1,filters))(sq)
            X = Multiply()([sq , X])
        return X
    
    def _resnet_block(self, inputs , filters , strides):
        if(strides != 1):
            identity = Conv2D(filters , kernel_size = 1 , strides = 2 , padding = 'same')(inputs)
        else:
            identity = inputs
        
        conv = Conv2D(filters , kernel_size = 3 , strides = strides , padding = 'same')(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        
        conv = Conv2D(filters , kernel_size = 3 , strides = 1 , padding = 'same')(conv)
        conv = BatchNormalization()(conv)
        
        output = Add()([identity , conv])
        output = Activation('relu')(output)
        
        sq = GlobalAveragePooling2D()(output)
        sq = Dense(int(filters / 4) , activation = 'relu')(sq)
        sq = Dense(filters , activation = 'sigmoid')(sq)
        sq = Reshape((1,1,filters))(sq)
        ex = Multiply()([sq , output])
        
        return ex
    
    def _upsample(self , inputs):
        X = UpSampling2D(size = (2,2))(inputs)
        return X
    def _build_model(self):
        inputs = Input(self.input_shape)
        s = Lambda(lambda x: x / 255) (inputs)
        
        conv1 = self._convlayer(s , filters = 8 , strides = 1 , kernel_size = 1)
        
        e1 = self._resnet_block(conv1 , filters = 8 , strides = 1)
        
        e2 = self._resnet_block(e1 , filters = 16 , strides = 2)
        
        e3 = self._resnet_block(e2 , filters = 32 , strides = 2)
        
        e4 = self._resnet_block(e3 , filters = 64 , strides = 2)
        
        e5 = self._resnet_block(e4 , filters = 128 , strides = 2)
        
        
        d1 = self._upsample(e5)
        d1 = self._convlayer(d1 , filters = 64 , kernel_size = 3 , strides = 1)
        d1 = self._convlayer(d1 , filters = 64 , kernel_size = 3 , strides = 1)
        d1 = self._convlayer(d1 , filters = 64 , kernel_size = 3 , strides = 1)
        
        d2 = self._upsample(d1)
        d2 = self._convlayer(d2 , filters = 32 , kernel_size = 3 , strides = 1)
        d2 = self._convlayer(d2 , filters = 32 , kernel_size = 3 , strides = 1)
        d2 = self._convlayer(d2 , filters = 32 , kernel_size = 3 , strides = 1)
        
        d3 = self._upsample(d2)
        d3 = self._convlayer(d3 , filters = 16 , kernel_size = 3 , strides = 1)
        d3 = self._convlayer(d3 , filters = 16 , kernel_size = 3 , strides = 1)
        
        d4 = self._upsample(d3)
        d4 = self._convlayer(d4 , filters = 8 , kernel_size = 3 , strides = 1)
        d4 = self._convlayer(d4 , filters = self.num_classes , kernel_size = 1 , strides = 1 , activation = 'softmax')
        
               
        
        model = Model(inputs = inputs , outputs = d4)
        
        return model