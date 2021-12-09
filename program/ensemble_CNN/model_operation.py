#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-26 10:45:13
# model_operation.py

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import callbacks

def model_load(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU})

class CopyWeights(callbacks.Callback):
    def __init__(self, model):
        self.count = 0
        self.model = model
        
    def on_train_batch_begin(self, batch, logs=None):# 1バッチの訓練が始まる際にすることを書く
        conv_weights = self.model.get_layer('conv2D1').get_weights()
        self.model.get_layer('conv2D2').set_weights(conv_weights)
        self.count += 1
        
    def on_train_end(self, logs=None):
        conv_weights = self.model.get_layer('conv2D1').get_weights()
        self.model.get_layer('conv2D2').set_weights(conv_weights)

# https://wak-tech.com/archives/1761
def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            # 重み，バイアスの初期化
            if ("kernel_initializer" or "recurrent_initializer") not in key: # 重みの初期値を探す
                  continue #if no, skip it
            else:
                # print("key:")
                # print(key)
                # print("kernel_initializer:")
                # print(initializer)
                
                # 既存の重み，バイアスを初期値に入れ替え
                weights = layer.get_weights()
                weights = [initializer(w.shape, w.dtype) for w in weights] 
                layer.set_weights(weights)
    return model