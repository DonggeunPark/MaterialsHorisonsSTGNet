# -*- coding: utf-8 -*-
"""
Created on Fri May 17 09:34:47 2024

@author: user
"""

#%%
# =============================================================================
# Model load step
# =============================================================================
from keras.models import model_from_json, load_model 

json_file = open("STGNet.json", "r") # 100_120_featuremap 128
loaded_model_json = json_file.read() 
json_file.close() 
model = tf.keras.models.model_from_json(loaded_model_json)
model.load_weights("STGNet.h5") 

grid = np.load('InputConfiguration.npy')
stress_pre = model.predict(grid)

