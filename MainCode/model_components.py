# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:40:19 2024

@author: user
"""
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, concatenate

def MB(inputs, channels):
    conv1_2 = Conv2D(channels//3, (3, 3), activation=None, padding='same')(inputs)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1_2 = Activation('relu')(conv1_2)
    
    conv1_3 = Conv2D(channels//3, (6, 6), activation=None, padding='same')(inputs)
    conv1_3 = BatchNormalization()(conv1_3)
    conv1_3 = Activation('relu')(conv1_3)
    
    conv1_4 = Conv2D(channels//3, (9, 9), activation=None, padding='same')(inputs)
    conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Activation('relu')(conv1_4)
    
    conv1 = add([conv1_2, conv1_3, conv1_4]) #chnnel[0]//3 , chnnel[1]//3
    conv2 = concatenate([inputs,conv1]) # 1 + channels[0]//3, 1 + channels[0]//3 + channels[0]//3 + chnnel[1]//3  
    
    conv2_1 = Conv2D(channels//3, (3, 3), activation=None, padding='same')(conv2)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_1 = Activation('relu')(conv2_1)
    
    conv2_2 = Conv2D(channels//3, (6, 6), activation=None, padding='same')(conv2)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    
    conv2_3 = Conv2D(channels//3, (9, 9), activation=None, padding='same')(conv2)
    conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    
    conv3 = add([conv2_1, conv2_2, conv2_3]) 
    result = concatenate([inputs, conv2, conv3]) 
    result = Conv2D(channels, (1, 1), activation=None, padding='same')(result)
    result = BatchNormalization()(result)
    result = Activation('relu')(result)
    
    return result