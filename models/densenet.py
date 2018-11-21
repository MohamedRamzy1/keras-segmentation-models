# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:46:45 2018

@author: Mohamed
"""

from keras.models import Model
from keras.layers import Input , Concatenate, BatchNormalization , Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import *
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.layers import Dropout

class DenseNet:
    def __init__(self , input_shape , num_classes , growth_rate):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.model = self._build_model()
        
        
    def _conv_block(self , inputs):
      X = BatchNormalization()(inputs)
      X = Activation('relu')(X)
      X = Conv2D(filters = self.growth_rate , kernel_size = 3 , kernel_initializer = he_normal(2019) , strides = 1 , padding = 'same')(X)
      X = Dropout(0.2)(X)
      return X
  
    def _dense_block(self,inputs , n_layers):
      layers = []
      
      identity = inputs
      
      for i in range(n_layers):
        
        X = self._conv_block(identity)
        layers.append(X)
        if(i != n_layers - 1):
          X = Concatenate()([X , identity])
          identity = X
           
      output = Concatenate()(layers)  
      return output
  
    def _transition_down(self , inputs , filters):
      X = BatchNormalization()(inputs)
      X = Activation('relu')(X)
      X = Conv2D(filters = filters , kernel_size = 1 ,  kernel_initializer = he_normal(2019) , strides = 1 , padding = 'same')(X)
      X = Dropout(0.2)(X)
      X = MaxPooling2D(pool_size = (2 , 2) , strides = 2)(X)
    
      return X
  
    def _transition_up(self , inputs , filters):
      X = Conv2DTranspose(filters = filters , kernel_size = 3 ,  kernel_initializer = he_normal(2019) , strides = 2 , padding = 'same')(inputs)
      return X
    
    def _build_model(self):
        #DenseNet model
    
        inp = Input(self.input_shape)
        
        conv1 = Conv2D(48 , kernel_size = 3 , kernel_initializer = he_normal(2019) , kernel_regularizer = l2(1e-4) , strides = 1 , padding = 'same')(inp)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        
        last_channel = 48
        
        db1 = self._dense_block(conv1 , 4)
        conc_db1 = Concatenate()([conv1 , db1])
        last_channel = 4 * self.growth_rate + last_channel
        td1 = self._transition_down(conc_db1 , last_channel)
        
        db2 = self._dense_block(td1 , 5)
        conc_db2 = Concatenate()([td1 , db2])
        last_channel = 5 * self.growth_rate + last_channel
        
        td2 = self._transition_down(conc_db2 , last_channel)
        
        db3 = self._dense_block(td2 , 7)
        conc_db3 = Concatenate()([td2 , db3])
        last_channel = 7 * self.growth_rate + last_channel
        
        td3 = self._transition_down(conc_db3 , last_channel)
        
        db4 = self._dense_block(td3 , 10)
        conc_db4 = Concatenate()([td3 , db4])
        last_channel = 10 * self.growth_rate + last_channel
        td4 = self._transition_down(conc_db4 , last_channel)
        
        db5 = self._dense_block(td4 , 12)
        conc_db5 = Concatenate()([td4 , db5])
        last_channel = 12 * self.growth_rate + last_channel
        
        td5 = self._transition_down(conc_db5 , last_channel)
        
        bn = self._dense_block(td5 , 15)
        
        
        tu1 = self._transition_up(bn , 15 * self.growth_rate)
        cup_1 = Concatenate()([tu1 , conc_db5])
        db6 = self._dense_block(cup_1 , 12)
        
        tu2 = self._transition_up(db6 , 12 * self.growth_rate)
        cup_2 = Concatenate()([tu2 , conc_db4])
        db7 = self._dense_block(cup_2 , 10)
        
        tu3 = self._transition_up(db7 , 10 * self.growth_rate)
        cup_3 = Concatenate()([tu3 , conc_db3])
        db8 = self._dense_block(cup_3 , 7)
        
        tu4 = self._transition_up(db8 , 7 * self.growth_rate)
        cup_4 = Concatenate()([tu4 , conc_db2])
        db9 = self._dense_block(cup_4 , 5)
        
        tu5 = self._transition_up(db9 , 5 * self.growth_rate)
        cup_5 = Concatenate()([tu5 , conc_db1])
        db10 = self._dense_block(cup_5 , 4)
        
        proj = Conv2D(filters = self.num_classes , kernel_size = 1 , kernel_initializer = he_normal(2019), kernel_regularizer = l2(1e-4) , strides = 1 , padding = 'same')(db10)
        
        output = Activation('softmax')(proj)
        
        model = Model(inputs = inp , output = output)
        
        model.summary()
        
        return model

