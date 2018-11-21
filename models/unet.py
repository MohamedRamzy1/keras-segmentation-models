# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 00:06:28 2018

@author: Mohamed
"""

from keras.models import Model, load_model
from keras.layers import *
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

class Unet:
    def __init__(self , input_shape , num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self):
        input_img = Input(self.input_shape, name='img')
        
        inp = BatchNormalization()(input_img) 
        
        c1 = Conv2D(4, (3, 3), activation='relu', padding='same') (inp)
        a1 = MaxPooling2D((2, 2))(c1)
        c1 = Dropout(0.2)(c1)
        c1 = Conv2D(4, (3, 3), activation='relu', padding='same') (c1)
        p1 = MaxPooling2D((2, 2)) (c1)
        
        cat1 = concatenate([p1, a1])
        
        c2 = Conv2D(8, (3, 3), activation='relu', padding='same') (cat1)
        a2 = MaxPooling2D((2, 2))(c2)
        c2 = Dropout(0.2)(c2)
        c2 = Conv2D(8, (3, 3), activation='relu', padding='same') (c2)
        p2 = MaxPooling2D((2, 2)) (c2)
        
        cat2 = concatenate([p2, a2])
        
        c3 = Conv2D(16, (3, 3), activation='relu', padding='same') (cat2)
        a3 = MaxPooling2D((2, 2))(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(16, (3, 3), activation='relu', padding='same') (c3)
        p3 = MaxPooling2D((2, 2)) (c3)
        
        cat3 = concatenate([p3, a3])
        
        c4 = Conv2D(32, (3, 3), activation='relu', padding='same') (cat3)
        a4 = MaxPooling2D((2, 2))(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(32, (3, 3), activation='relu', padding='same') (c4)
        p4 = MaxPooling2D((2, 2)) (c4)
        
        cat4 = concatenate([p4, a4])
        
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (cat4)
        a5 = MaxPooling2D((2, 2))(c5)
        c5 = Dropout(0.2)(c5)
        c5 = Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
        p5 = MaxPooling2D((2, 2)) (c5)
        
        cat5 = concatenate([p5, a5])
        
        c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (cat5)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', padding='same') (c6)
        
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c5])
        c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', padding='same') (c7)
        
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c4])
        c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (u8)
        c8 = Dropout(0.2)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', padding='same') (c8)
        
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c3])
        c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (u9)
        c9 = Dropout(0.2)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', padding='same') (c9)
        
        u10 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c9)
        u10 = concatenate([u10, c2])
        c10 = Conv2D(8, (3, 3), activation='relu', padding='same') (u10)
        c10 = Dropout(0.2)(c10)
        c10 = Conv2D(8, (3, 3), activation='relu', padding='same') (c10)
        
        u11 = Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same') (c10)
        u11 = concatenate([u11, c1], axis = 3)
        c11 = Conv2D(4, (3, 3), activation='relu', padding='same') (u11)
        c11 = Dropout(0.2)(c11)
        c11 = Conv2D(4, (3, 3), activation='relu', padding='same') (c11)
        
        outputs = Conv2D(self.num_classes, (1, 1), activation='softmax') (c11)
        
        model = Model(inputs=[input_img], outputs=[outputs])
        
        return model