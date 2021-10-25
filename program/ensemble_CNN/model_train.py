#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-25 20:32:46
# model_train.py

import tensorflow as tf
import numpy as np
import os
import math
import gc

from tensorflow.keras.utils import Sequence

from read_data import read_ymemmap

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

def model_train(model, 
                verbose, 
                epochs, 
                batch_size, 
                callbacks, 
                load_split_batch, 
                memmap_dir, 
                y_dim, 
                output_num, 
                output_axis, 
                data_size
                ):
    if not load_split_batch:
        #訓練データを外から中に入れる
        x_train_data = np.memmap(filename=os.path.join(memmap_dir, 'x_train_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
        y_train_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_train_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
        x_val_data = np.memmap(filename=os.path.join(memmap_dir, 'x_val_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
        y_val_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_val_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
        
        history = model.fit([x_train_data[:,:,:,0],x_train_data[:,:,:,1]],
                            y_train_data,
                            verbose=verbose,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=([x_val_data[:,:,:,0], x_val_data[:,:,:,1]], y_val_data),
                            callbacks=callbacks,
                            )
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

# 出力の絶対値平均を計算
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