#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-26 10:45:13
# model_operation.py

import sys
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import callbacks

from my_callbacks import CopyWeights

def model_load(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU, 'CopyWeights': CopyWeights})

# グレースケール前提
def check_input_shape(model):
    # tensorflow inputの見分け方
    #   1入力はタプルで囲まれる．ex. (None, 32, 32, 2)
    #   複数入力はタプルで囲まれたものが入力分あり，それがリストの中に入っている．
    #                          ex. [(None, 32, 32), (None, 32, 32)]
    #   model.input_shapeでインスタンス化したモデルの入力の形を確認できる．
    
    input_shape = model.input_shape
    if isinstance(input_shape, tuple): # タプルかどうか判定
        input_num = 1 # 入力は1つ
        input_depth = input_shape[-1] # 画像の深さ方向は -1 番目の要素
        return input_num, input_depth
    elif isinstance(input_shape, list): # リストかどうか判定
        input_num = len(input_shape) # 入力はリストの要素数
        input_depth = 1 # 画像の深さ方向は1
        return input_num, input_depth
    print('{} : 入力の形状を確認できませんでした．'.format(model))
    sys.exit(1) # 異常終了

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