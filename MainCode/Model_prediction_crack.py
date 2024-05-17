# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:50:19 2024

@author: user
"""

#%%
# =============================================================================
# Model load step
# =============================================================================
from keras.models import model_from_json, load_model 

json_file = open("STGNet-crack.json", "r") # 100_120_featuremap 128
loaded_model_json = json_file.read() 
json_file.close() 
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("STGNet-crack.h5") 

grid = np.load('InputConfiguration.npy')
crack_pre = model.predict(grid)
