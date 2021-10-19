#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:44:31
# my_model.py

import tensorflow as tf
import numpy as np
import os
import math

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.metrics import mean_absolute_percentage_error
from tensorflow.keras import callbacks
from tensorflow.keras.utils import Sequence

from read_data import read_ymemmap

class MySequence(Sequence): # generator
    def __init__(self, data_size, batch_size, file_name=None, memmap_dir=None, y_dim=None, output_num=None, output_axis=None):
        # memmap用のファイルのパス
        X_MEMMAP_PATH = 'x_' + file_name + '.npy'
        Y_MEMMAP_PATH = 'y_' + file_name + '.npy'
        self.batch_size = batch_size
        self.length = math.ceil(data_size / batch_size)
        self.memmap_X = np.memmap(
            filename=os.path.join(memmap_dir, X_MEMMAP_PATH), dtype=np.float32, mode='r',
            shape=(data_size, 32, 32, 2))
        self.memmap_y = read_ymemmap(filename=os.path.join(memmap_dir, Y_MEMMAP_PATH), dtype=np.float32, mode='r', y_dim=y_dim, output_num=output_num, output_axis=output_axis)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.memmap_X[start_idx:last_idx]
        y = self.memmap_y[start_idx:last_idx]
        
        return X, y

    def __len__(self):
        # 全データの長さ
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass

class MySequenceF(Sequence): # generator
    def __init__(self, data_size, batch_size, file_name=None, memmap_dir=None, y_dim=None, output_num=None, output_axis=None):
        # memmap用のファイルのパス
        X_MEMMAP_PATH = 'x_' + file_name + '.npy'
        Y_MEMMAP_PATH = 'y_' + file_name + '.npy'
        self.batch_size = batch_size
        self.length = math.ceil(data_size / batch_size)
        self.memmap_X = np.memmap(
            filename=os.path.join(memmap_dir, X_MEMMAP_PATH), dtype=np.float32, mode='r',
            shape=(data_size, 32, 32, 2))
        self.memmap_y = read_ymemmap(filename=os.path.join(memmap_dir, Y_MEMMAP_PATH), dtype=np.float32, mode='r', y_dim=y_dim, output_num=output_num, output_axis=output_axis)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.memmap_X[start_idx:last_idx]
        y = self.memmap_y[start_idx:last_idx]
        
        return [X[:,:,:,0], X[:,:,:,1]], y

    def __len__(self):
        # 全データの長さ
        return self.length

    def on_epoch_end(self):
        # epoch終了時の処理
        pass

# Sequentialモデルを生成する関数，掲示板参照
def my_sequential_model_builder(layers = None, optimizer = 'rmsprop', loss = None, metrics = None):
# layersのフォーマット:
#   ( # <- tuple!
#    (レイヤー名1, キーワードなし引数の値, "キーワード付き引数の値を表す辞書の文字列"), # <- tuple!
#    (レイヤー名2, キーワードなし引数の値, "キーワード付き引数の値を表す辞書の文字列"), # <- tuple!
#     ...
#   ) # <- tuple!
#
# example:
# (
#  (tf.keras.layers.Dense, 5, "{'activation': 'relu', 'input_value': (3,)}"),
#  (tf.keras.layers.Dense, "{'units': 6, 'activation': 'relu'}"),
#  (tf.keras.layers.Dense, json.dumps(dict(units = 6, activation = 'relu'))),
#  (tf.keras.layers.Dense, "{'units': 2"}),
#   ...
# )
    model = tf.keras.models.Sequential()
    if layers is None:
        layers = tuple() # empty tuple
    elif not hasattr(layers, '__iter__'):
        layers = (layers,)
    for i in layers:
        if i is None:
            continue
        elif not hasattr(i, '__iter__'):
            i = (i,)
        layer = None
        args = []
        kwargs = {}
        for j in i:
            if isinstance(j, type):
                layer = j
            elif isinstance(j, dict):
                kwargs.update(j)
            elif isinstance(j, str) and j.strip().startswith('{'):
                kwargs.update(ast.literal_eval(j))
            elif hasattr(j, '__iter__'):
                args.extend(j)
            else:
                args.append(j)
        if layer is MaxPooling2D:
            a = kwargs.pop('switch')
            if a == 'off':
                continue
        elif layer is Dense or layer is Dropout:
            if None in args or None in kwargs.values():
                continue
        model.add(layer(*args, **kwargs))
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    return model

def model_train(model, verbose, epochs, batch_size, callbacks, load_split_batch, model_type):
    if not load_split_batch:
        #訓練データを外から中に入れる
        x_train_copy = np.copy(x_train_data)
        y_train_copy = np.copy(y_train_data)
        x_val_copy = np.copy(x_val_data)
        y_val_copy = np.copy(y_val_data)
        
        if model_type == 'Sequential':
            history = model.fit(x_train_copy,
                                y_train_copy,
                                verbose=verbose,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_val_copy, y_val_copy),
                                callbacks=callbacks,
                                )
        elif model_type == 'functional_API':
            history = model.fit([x_train_copy[:,:,:,0],x_train_copy[:,:,:,1]],
                                y_train_copy,
                                verbose=verbose,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=([x_val_copy[:,:,:,0], x_val_copy[:,:,:,1]], y_val_copy),
                                callbacks=callbacks,
                                )
        
        del x_train_copy, y_train_copy
    else:
        #データ数を外から中に入れる
        train_data_size = data_size[0]
        val_data_size = data_size[1]
        
        # バッチごとにデータの読み込み
        if model_type == 'Sequential':
            history = model.fit(x=MySequence(train_data_size, batch_size, 'train_data', memmap_dir, y_dim, output_num, output_axis),
                                verbose=verbose,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=MySequence(val_data_size, batch_size, 'val_data', memmap_dir, y_dim, output_num, output_axis),
                                )
        elif model_type == 'functional_API':
            history = model.fit(x=MySequenceF(train_data_size, batch_size, 'train_data', memmap_dir, y_dim, output_num, output_axis),
                                verbose=verbose,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=MySequenceF(val_data_size, batch_size, 'val_data', memmap_dir, y_dim, output_num, output_axis),
                                )
            
        del train_data_size, val_data_size