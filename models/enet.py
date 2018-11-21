# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:09:51 2018

@author: Mohamed
"""
import tensorflow as tf
from keras.layers import Conv2D , UpSampling2D , Conv2DTranspose , MaxPooling2D , PReLU , Input , BatchNormalization , Concatenate , Add , Lambda , SpatialDropout2D , Reshape , Activation
from keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint
from keras.models import Model , load_model
from keras.regularizers import l2
from keras.optimizers import Adam , SGD
from keras.metrics import categorical_accuracy

class ENet:
    def __init__(self , input_shape , num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        

    def _conv_block(self , inp , filters , kernel_size , strides = 1 , dilation = 1):
      C = Conv2D(filters = filters , kernel_size = kernel_size , use_bias = False , activation = None , strides = strides 
                 , padding = 'same' ,
                 dilation_rate = dilation)(inp)
      C = BatchNormalization()(C)
      C = PReLU(shared_axes = [1,2])(C)
      
      return C

    def _deconv_block(self , inp , filters):
      C = Conv2DTranspose(filters = filters , kernel_size = 3 , use_bias = False , activation = None , strides = 2 , padding = 'same')(inp)
      C = BatchNormalization()(C)
      C = PReLU(shared_axes = [1,2])(C)
      
      return C

    def _initial_block(self , inp):
      
      C = self._conv_block(inp , filters = 13 , kernel_size = 3 , strides = 2)
      
      M = MaxPooling2D(pool_size = 3 , strides = 2 , padding = 'same')(inp)
      
      out = Concatenate()([C , M])
      
      return out

    def _make_zeros(self, to_pad):
      def output(t):
        return t[: , : , : , :to_pad]
      return output

    def _bottleneck(self , inp , filters  , dilation = 1 , asymmetric = False , sampling = None):
      
      reduced_size = int(int(inp.shape[3]) / 4)
      if reduced_size == 0:
        reduced_size = filters
      if sampling == 'down':
        C = self._conv_block(inp , filters = reduced_size , kernel_size = 2 , strides = 2)
      
      else:
        C = self._conv_block(inp , filters = reduced_size , kernel_size = 1)
      
      if asymmetric:
        C = self._conv_block(C , filters = reduced_size , kernel_size = (5,1))
        C = self._conv_block(C , filters = reduced_size , kernel_size = (1,5))
        
      if sampling == 'up':
        C = self._deconv_block(C , filters = reduced_size)
      
      elif sampling != 'up' and not asymmetric:
        C = self._conv_block(C , filters = reduced_size , kernel_size = 3 , dilation = dilation)
        
        
      C = Conv2D(filters = filters , kernel_size = 1 , use_bias = False , activation = None , strides = 1 , padding = 'same')(C)
      if sampling == None:
        C = BatchNormalization()(C)
        C = PReLU(shared_axes = [1,2])(C)
        return C
      
      elif sampling == "down":
        M = MaxPooling2D(pool_size = 2 , padding = 'same' , strides = 2)(inp)
        to_pad = filters - int(inp.shape[3])
        zeros_tensor1 = Lambda(tf.zeros_like)(M)
        zeros_tensor2 = Lambda(tf.zeros_like)(M)
        zeros_tensor3 = Lambda(tf.zeros_like)(M)
        zeros_tensor4 = Lambda(tf.zeros_like)(M)
        
        zeros_tensor = Concatenate()([zeros_tensor1 , zeros_tensor2 , zeros_tensor3 , zeros_tensor4])
    
        zeros_tensor = Lambda(self._make_zeros(to_pad))(zeros_tensor)
        M = Concatenate()([M , zeros_tensor])
        output = Add()([M , C])
        output = BatchNormalization()(output)
        output = PReLU(shared_axes = [1,2])(output)
        
      elif sampling == "up":
        U = UpSampling2D(size = (2,2))(inp)
        U = Conv2D(filters = filters , kernel_size = 3 , use_bias = False , activation = None , strides = 1 , padding = 'same')(U)
        U = BatchNormalization()(U)
        output = Add()([U , C])
        output = BatchNormalization()(output)
        output = PReLU(shared_axes = [1,2])(output)
      
      return output
    
    def build_model(self):
        inp = Input(self.input_shape)
        init_block = self._initial_block(inp)
        
        stage1 = self._bottleneck(init_block , filters = 64 , sampling = 'down')
        for _ in range(4):
          stage1 = self._bottleneck(stage1 , filters = 64)
          
        stage2 = self._bottleneck(stage1 , filters = 128 , sampling = 'down')
        stage2 = self._bottleneck(stage2 , filters = 128)
        stage2 = self._bottleneck(stage2 , filters = 128 , dilation = 2)
        stage2 = self._bottleneck(stage2 , filters = 128 , asymmetric = True)
        stage2 = self._bottleneck(stage2 , filters = 128 , dilation = 4)
        stage2 = self._bottleneck(stage2 , filters = 128)
        stage2 = self._bottleneck(stage2 , filters = 128 , dilation = 8)
        stage2 = self._bottleneck(stage2 , filters = 128 , asymmetric = True)
        stage2 = self._bottleneck(stage2 , filters = 128 , dilation = 16)
        
        stage3 = self._bottleneck(stage2 , filters = 128)
        stage3 = self._bottleneck(stage3 , filters = 128 , dilation = 2)
        stage3 = self._bottleneck(stage3 , filters = 128 , asymmetric = True)
        stage3 = self._bottleneck(stage3 , filters = 128 , dilation = 4)
        stage3 = self._bottleneck(stage3 , filters = 128)
        stage3 = self._bottleneck(stage3 , filters = 128 , dilation = 8)
        stage3 = self._bottleneck(stage3 , filters = 128 , asymmetric = True)
        stage3 = self._bottleneck(stage3 , filters = 128 , dilation = 16)
        
        stage4 = self._bottleneck(stage3 , filters = 64 , sampling = 'up')
        stage4 = self._bottleneck(stage4 , filters = 64)
        stage4 = self._bottleneck(stage4 , filters = 64)
        
        stage5 = self._bottleneck(stage4 , filters = 16 , sampling = 'up')
        stage5 = self._bottleneck(stage5 , filters = 16)
        
        output = Conv2DTranspose(filters = self.num_classes , kernel_size = 3 , activation = 'softmax' , strides = 2 , padding = 'same')(stage5)
        
        model = Model(inputs = inp , outputs = output)
        return model
