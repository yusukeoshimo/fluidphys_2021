#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:44:31
# my_model.py

import tensorflow as tf
import numpy as np
import os
import math
import gc
import ast

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

def model_train(model, verbose, epochs, batch_size, callbacks, load_split_batch, memmap_dir, y_dim, output_num, output_axis, data_size):
    if not load_split_batch:
        #訓練データを外から中に入れる
        x_train_data = np.memmap(filename=os.path.join(memmap_dir, 'x_train_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
        y_train_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_train_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
        x_val_data = np.memmap(filename=os.path.join(memmap_dir, 'x_val_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
        y_val_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_val_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
        
        input_data = [x_train_data[:,:,:,0],x_train_data[:,:,:,1]]
        val_input_data = [x_val_data[:,:,:,0], x_val_data[:,:,:,1]]
        
        history = model.fit(input_data,
                            y_train_data,
                            verbose=verbose,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(val_input_data, y_val_data),
                            callbacks=callbacks,
                            )
        
        del x_train_data, y_train_data
    else:
        #データ数を外から中に入れる
        train_data_size = data_size[0]
        val_data_size = data_size[1]
        
        # バッチごとにデータの読み込み
        
        history = model.fit(x=MySequenceF(train_data_size, batch_size, 'train_data', memmap_dir, y_dim, output_num, output_axis),
                            verbose=verbose,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=MySequenceF(val_data_size, batch_size, 'val_data', memmap_dir, y_dim, output_num, output_axis),
                            )
            
        del train_data_size, val_data_size
    return model, history

def calc_am(memmap_dir, y_dim, output_num, output_axis):
    memmap_list = os.listdir(memmap_dir)
    for i in memmap_list:
        # 1/N Σ(|x|+|y|+|z|)/3
        if 'y_' in i:
            if 'train' in i:
                y_memmap = read_ymemmap(filename=os.path.join(memmap_dir, i), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
                train_am = np.mean(np.mean(np.abs(y_memmap),axis=-1))
            elif 'val' in i:
                y_memmap = read_ymemmap(filename=os.path.join(memmap_dir, i), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
                val_am = np.mean(np.mean(np.abs(y_memmap),axis=-1))
            elif 'test' in i:
                y_memmap = read_ymemmap(filename=os.path.join(memmap_dir, i), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
                test_am = np.mean(np.mean(np.abs(y_memmap),axis=-1))
    del y_memmap
    gc.collect()
    return train_am, val_am, test_am
