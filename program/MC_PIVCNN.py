#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-09-29 15:49:06
# MC_PIVCNN.py

#---- オプション ----
#-d[irectory] dir_path         -> 訓練データがあるディレクトリを指定．zipファイルでも可．memmapファイルの入ったディレクトリでも可
#-r[emove]                     -> 最適化の前回までのデータがある場合，それを消す．
#-b[atch]                      -> 訓練データをバッチごとに読み込む．
#-o[utput_num] number (axis)   -> 出力の数を指定，出力の数が1の場合は軸（x, y, z）も指定．
#-i[nitial]                    -> 最適化の初期値を固定して実行．
#-m[onte]                      -> モンテカルロドロップアウト（予測時にもドロップアウト層を使うような学習モデルの作成）．
#-n n_jobs                     -> 並列処理の分割数をn_jobs個に設定．
#-t[ime] time_out              -> [hour]最適化実行時間，time_out時間実行．
#-c[onfirmation]               -> 最適なハイパーパラメータの確認．
#-h[elp]                       -> ヘルプの表示．

#デフォルトの設定
#訓練データの読み込み設定  : すべて読み込む
#前回までの最適化データ    : 前回の続きから開始する
#並列計算における分割数   : シングルコアで実行
#データの保存場所         : 作業フォルダ
#出力数                 : 2
#ドロップアウト層の値     : 0.5

import sys
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
import cv2 as cv
import gc
import optuna
import math
from glob import glob
from natsort import natsorted
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from adjustText import adjust_text
import multiprocessing
from joblib import Parallel, delayed
import zipfile
import matplotlib

#バックエンドを指定
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import mean_absolute_percentage_error
from tensorflow.keras import callbacks
from tensorflow.keras import backend
from tensorflow.keras.utils import Sequence

def mode_detect():
    try:
      tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
      tpu = None
      gpus = tf.config.experimental.list_logical_devices("GPU")

    if tpu:
      tf.tpu.experimental.initialize_tpu_system(tpu)
      strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
      print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
      strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
      print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
      strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
      print('Running on single GPU ', gpus[0].name)
    else:
      strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
      print('Running on CPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return strategy

#動的グラフを描画するためのクラス，ネット参照
class LossHistory(callbacks.Callback):
    def __init__(self, save_name=None, trial_num=None, loss='loss', metrics=None, minimize_loss=None, am_list=None, output_num=None, memmap_dir=None):
        # コンストラクタに保持用の配列を宣言しておく
        self.save_name = save_name
        self.trial_num = trial_num
        self.loss = loss
        self.metrics = metrics
        self.minimize_loss = minimize_loss
        self.memmap_dir = memmap_dir
        if output_num == 2:
            self.ylabel = 'loss/$1/N\sum_{k=1}^{N} (|x_k|+|y_k|)/$' + str(output_num)
        else:
            self.ylabel = 'loss/$1/N\sum_{k=1}^{N} (|x_k|+|y_k|+|z_k|)/$' + str(output_num)
        if hasattr(am_list, '__iter__'):
            self.train_am = am_list[0]
            self.val_am = am_list[1]
            self.test_am = am_list[2]
        self.train_loss = []
        self.train_metrics = []
        self.val_loss = []
        self.val_metrics = []

    def on_epoch_end(self, epoch, logs={}):
        # 配列にepochが終わるたびにAppendしていく
        self.train_loss.append(logs[self.loss])
        self.train_metrics.append(logs[self.loss]/self.train_am)
        self.val_loss.append(logs['val_'+self.loss])
        self.val_metrics.append(logs['val_'+self.loss]/self.val_am)
    
    def on_train_end(self, logs={}):
        if self.minimize_loss > logs['val_'+self.loss]:
            self.minimize_loss = logs['val_'+self.loss]
            # loss_history.txtにデータを保存
            array_epoch = np.arange(len(self.train_loss)).reshape(-1, 1)
            array_train_loss = np.array(self.train_loss).reshape(-1, 1)
            array_val_loss = np.array(self.val_loss).reshape(-1, 1)
            array_train_metrics = np.array(self.train_metrics).reshape(-1, 1)
            array_val_metrics = np.array(self.val_metrics).reshape(-1, 1)
            loss_history = np.hstack((array_epoch, array_train_loss, array_val_loss, array_train_metrics, array_val_metrics))
            with open('loss_history.txt', 'w') as f:
                f.write(self.memmap_dir + '\n')
                f.write('# epochs                 train_loss               val_loss                 train_metrics            val_metrics\n')
                np.savetxt(f, loss_history)
            
            fig = plt.figure(num='epoch - loss', clear=True)
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.titlesize'] = 12.0
            plt.rcParams['axes.labelsize'] = 12.0
            plt.rcParams['axes.formatter.limits'] = [-3, 4] # [m, n] -> Scientific notation is used for data < 10^m or 10^n <= data
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['xtick.major.size'] = 8.0
            plt.rcParams['xtick.minor.size'] = 4.0
            plt.rcParams['xtick.labelsize'] = 12.0
            plt.rcParams['ytick.direction'] = 'in'
            plt.rcParams['ytick.major.size'] = 8.0
            plt.rcParams['ytick.minor.size'] = 4.0
            plt.rcParams['ytick.labelsize'] = 12.0
            plt.rcParams['legend.edgecolor'] = 'k'
            plt.rcParams['legend.fontsize'] = 12.0

            # グラフ描画部
            plt.title('learning curve trial: '+str(self.trial_num))
            plt.xlabel('epoch')
            plt.ylabel('loss/$1/N\sum_{k=1}^{N} (|x_k|+|y_k|)/2$')
            plt.plot(self.train_metrics, label=self.loss)
            plt.plot(self.val_metrics, label='val_'+self.loss)
            
            # lossの最小値を求める
            train_metrics_best_score = np.nanmin(self.train_metrics)
            train_metrics_best_epochs = np.nanargmin(self.train_metrics)
            val_metrics_best_score = np.nanmin(self.val_metrics)
            val_metrics_best_epochs = np.nanargmin(self.val_metrics)
            train_loss_best_score = np.nanmin(self.train_loss)
            val_loss_best_score = np.nanmin(self.val_loss)
            # 縦線
            plt.axvline(train_metrics_best_epochs, color='b', linestyle=':')
            plt.axvline(val_metrics_best_epochs, color='r', linestyle=':')
            # テキスト
            text1 = plt.text(train_metrics_best_epochs, np.nanmax(self.train_metrics),'epoch:{}, train_loss:{:.6f}, train_metrics:{:.6f}'.format(train_metrics_best_epochs, train_loss_best_score, train_metrics_best_score), ha='right')
            text2 = plt.text(val_metrics_best_epochs, np.nanmax(self.val_metrics),'epoch:{}, val_loss:{:.6f}, val_metrics:{:.6f}'.format(val_metrics_best_epochs, val_loss_best_score, val_metrics_best_score), ha='right')
            texts = [text1, text2]
            adjust_text(texts) # テキスト位置の整理
            plt.legend()
            
            fig.savefig(self.save_name + '.png', bbox_inches="tight", pad_inches=0.05)
        # 訓練終了時にリストを初期化
        self.train_metrics.clear()
        self.val_metrics.clear()
        self.train_loss.clear()
        self.val_loss.clear()
        plt.close()

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

# 画像をnumpy配列で読み出す関数，掲示板参照
def read_image(file_name, flags = cv.IMREAD_GRAYSCALE, resize = None, show = False):
    # flags = cv.IMREAD_COLOR | cv.IMREAD_GRAYSCALE | ...
    # (see https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80)
    im = cv.imread(file_name, flags)
    if flags == cv.IMREAD_COLOR:
        im = im[:, :, (2, 1, 0)] # BGR -> RGB
    if resize is not None:
        if type(resize[0]) is int and type(resize[1]) is int:
            im = cv.resize(im, dsize = resize)
        else:
            im = cv.resize(im, dsize = None, fx = resize[0], fy = resize[1])
    if show:
        plt.clf()
        plt.imshow(im)
        plt.show()
    return im

# テキストファイルから入力データ，出力データを取り出す関数，掲示板参照
def get_input_output_from_file(file_name, top_skip = 0, input_columns = (0,), output_columns = (1,),
    delimiter = None, encoding = 'UTF-8'):
    # encoding = 'UTF-8' | 'CP932'
    if input_columns is None:
        input_columns = tuple() # empty tuple
    elif not hasattr(input_columns, '__iter__'):
        input_columns = (input_columns,)
    if output_columns is None:
        output_columns = tuple() # empty tuple
    elif not hasattr(output_columns, '__iter__'):
        output_columns = (output_columns,)
    input = []
    output = []
    with open(file_name, 'r', encoding = encoding) as f:
        for i in range(top_skip):
            next(f)
        for line in f:
            line = line.split(delimiter)
            input.append([float(line[i]) for i in input_columns])
            output.append([float(line[i]) for i in output_columns])
    input = np.array(input, dtype = 'float32')
    output = np.array(output, dtype = 'float32')
    if input.shape[-1] == 1:
        input = input.reshape(input.shape[:-1])
    if output.shape[-1] == 1:
        output = output.reshape(output.shape[:-1])
    return output

def read_ymemmap(filename, dtype=np.float32, mode='r', y_dim=3, output_num=2, output_axis=0):
    y_memmap = np.memmap(filename=filename, dtype=dtype, mode=mode).reshape(-1, y_dim)
    return y_memmap[:, : output_num] if output_num != 1 else y_memmap[:, output_axis].reshape(-1,1)

class MyDataset:
    def __init__(self, y_dim=None):
        self.y_dim = y_dim

    def im2array(self, dir_name, n_jobs=1):
        # example
        # 画像1の形状:[[ 0  1  0 ...  0  0  0]
        #             [ 6 10  4 ...  0  0  0]
        #             [30 46 20 ...  0  0  1]
        #             ...
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]]
        # 画像2の形状:[[ 0  0  0 ... 33  8  1]
        #             [ 0  0  0 ... 55 13  1]
        #             [ 0  0  0 ... 28  7  0]
        #             ...
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]]
        # 配列の結合:[[[ 0  0]
        #             [ 1  0]
        #             [ 0  0]
        #             ...
        #             [ 0 33]
        #             [ 0  8]
        #             [ 0  1]]
        #            [[ 6  0]
        #             [10  0]
        #             [ 4  0]
        #             ...
        #             [ 0 55]
        #             [ 0 13]
        #             [ 0  1]]
        #             ...   ]]]
        
        array_list = []
        dir_list = natsorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
        # dir_list: ['0', '1', '2', '3', ..., 'n']
        for i in tqdm(dir_list, desc='入力データ読み込み 全体の進捗'):
            file_count = sum(os.path.isfile(os.path.join(dir_name, i, name)) for name in os.listdir(os.path.join(dir_name, i)))
            a = Parallel(n_jobs=n_jobs, backend='threading')([delayed(self.read_process)(dir_name, i, j) for j in range(file_count//2)])
            array_list.extend(a)
            del a
        input_data = np.array(array_list)
        return input_data

    def read_process(self, dir_name, i, j):
        file_path_origin = os.path.join(dir_name,i,'origin_{}_{}.png'.format(i,j))
        file_path_next = os.path.join(dir_name,i,'next_{}_{}.png'.format(i,j))
        a = read_image(file_path_origin)
        b = read_image(file_path_next)
        return np.stack([a,b],-1)

    def save_memmap(self, data, memmap_dir):
        if not isinstance(data, tuple):
            data = (data,)
        for i in tqdm(data, desc='memmap作成 全体の進捗'):
            for key, value in globals().items():
                if id(i) == id(value):
                    file_name = key
            # memmap用のファイルのパス。
            MEMMAP_PATH = file_name + '.npy'

            data_size = len(i)
            if 'x' in file_name:
                memmap = np.memmap(filename=os.path.join(memmap_dir,MEMMAP_PATH), dtype=np.float32, mode='w+', shape=(data_size, 32, 32, 2))
            elif 'y' in file_name:
                memmap = np.memmap(filename=os.path.join(memmap_dir,MEMMAP_PATH), dtype=np.float32, mode='w+', shape=(data_size, self.y_dim))
            memmap[:] = i[:]
            del memmap

class MyGeneratorDataset:
    def __init__(self, y_dim=None, n_jobs=1):
        self.y_dim = y_dim
        self.n_jobs = n_jobs

    def get_paths(self, dir_name):
        self.dir_name = dir_name
        all_input_paths = []

        dir_list = natsorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
        for i in tqdm(dir_list, desc='訓練データ読み込み 進捗'):
            
            file_count = sum(os.path.isfile(os.path.join(dir_name, i, name)) for name in os.listdir(os.path.join(dir_name, i)))
            input_paths = Parallel(n_jobs=self.n_jobs, backend='threading')([delayed(self.append_input_path)(i, j) for j in range(file_count//2)])
            all_input_paths.extend(input_paths)
            del input_paths
        output_path = glob(os.path.join(dir_name,'**', 'dataset_output.txt'), recursive=True)[0]
        output_data = get_input_output_from_file(output_path, top_skip = 1, input_columns = None, output_columns = range(self.y_dim), delimiter = None, encoding = 'UTF-8')
        data_set = [[a, b] for a, b in zip(all_input_paths, output_data.tolist())]
        # data_setはリスト，中もリスト
        # data_set:[[[origin_0_0.png, next_0_0.png], [v_x0, v_y0]]
        #           [[origin_0_1.png, next_0_1.png], [v_x1, v_y1]]
        #           [[origin_0_2.png, next_0_2.png], [v_x2, v_y2]]
        #           ...
        #           [[origin_i_j.png, next_i_j.png], [v_xk, v_yk]]]
        return data_set

    def append_input_path(self, i, j):
        file_path_origin = os.path.join(self.dir_name, i, 'origin_{}_{}.png'.format(i,j))
        file_path_next = os.path.join(self.dir_name, i, 'next_{}_{}.png'.format(i,j))
        return [file_path_origin, file_path_next]

    def generate_memmap(self, data, memmap_dir):
        if not isinstance(data, tuple):
            data = (data,)
        size_list = []
        for i in tqdm(data, desc='memmap作成 全体の進捗'):
            size_list.append(len(i))
            for key, value in globals().items():
                if id(i) == id(value):
                    file_name = key
            # memmap用のファイルのパス。
            self.X_MEMMAP_PATH = 'x_' + file_name + '.npy'
            self.Y_MEMMAP_PATH = 'y_' + file_name + '.npy'

            self.data_size = len(i)
            X_memmap = np.memmap(
                filename=os.path.join(memmap_dir,self.X_MEMMAP_PATH), dtype=np.float32, mode='w+', shape=(self.data_size, 32, 32, 2))
            y_memmap = np.memmap(
                filename=os.path.join(memmap_dir,self.Y_MEMMAP_PATH), dtype=np.float32, mode='w+', shape=(self.data_size, self.y_dim))
            del X_memmap, y_memmap # memmap閉じる
            
            a = Parallel(n_jobs=self.n_jobs, backend='threading')([delayed(self.append_memmap)(memmap_dir, data, j) for data, j in zip(i, range(len(i)))])
            del a
            
        return size_list

    def append_memmap(self, memmap_dir, d, j):
        x = d[0]
        y = d[1]
        # xのデータをmemmapのファイルへ書き込む。
        a = read_image(x[0])
        b = read_image(x[1])
        X = np.stack([a,b],-1)
        
        X_memmap = np.memmap(
                filename=os.path.join(memmap_dir, self.X_MEMMAP_PATH), dtype=np.float32, mode='r+', shape=(self.data_size, 32, 32, 2))
        y_memmap = np.memmap(
                filename=os.path.join(memmap_dir, self.Y_MEMMAP_PATH), dtype=np.float32, mode='r+', shape=(self.data_size, self.y_dim))
        X_memmap[j] = X
        # example X
        # 画像1の形状:[[ 0  1  0 ...  0  0  0]
        #           [ 6 10  4 ...  0  0  0]
        #           [30 46 20 ...  0  0  1]
        #           ...
        #           [ 0  0  0 ...  0  0  0]
        #           [ 0  0  0 ...  0  0  0]
        #           [ 0  0  0 ...  0  0  0]]
        # 画像2の形状:[[ 0  0  0 ... 33  8  1]
        #           [ 0  0  0 ... 55 13  1]
        #           [ 0  0  0 ... 28  7  0]
        #           ...
        #           [ 0  0  0 ...  0  0  0]
        #           [ 0  0  0 ...  0  0  0]
        #           [ 0  0  0 ...  0  0  0]]
        # 配列の結合:[[[ 0  0]
        #           [ 1  0]
        #           [ 0  0]
        #           ...
        #           [ 0 33]
        #           [ 0  8]
        #           [ 0  1]]
        #          [[ 6  0]
        #           [10  0]
        #           [ 4  0]
        #           ...
        #           [ 0 55]
        #           [ 0 13]
        #           [ 0  1]]
        #           ...   ]]]
        
        # yのデータをmemmapのファイルへ書き込む。
        y_memmap[j] = np.array(y)
        # Y:[v_xk, v_yk]
        del X_memmap, y_memmap

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

def my_loss_function(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)/100

class Objective:
    def __init__(self, strategy, input_shape, y_dim, output_num, output_axis, memmap_dir, minimize_loss, load_split_batch, am_list, dropout_rate, set_initial_parms, model_type, best_model_name):
        self.strategy = strategy
        self.input_shape = input_shape
        self.y_dim = y_dim
        self.output_num = output_num
        self.output_axis = output_axis
        self.memmap_dir = memmap_dir
        self.minimize_loss = minimize_loss
        self.load_split_batch = load_split_batch
        self.am_list = am_list
        self.dropout_rate = dropout_rate
        self.set_initial_parms = set_initial_parms
        self.model_type = model_type
        self.best_model_name = best_model_name
        self.opt_figure = LossHistory()

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        keras.backend.clear_session()

        #最適化するパラメータの設定
        
        max_num_layer = 6 # Dense層の最大数
        l2 = 0.0001 # L2正則化の係数
        
        if trial._trial_id-1 == 0 and set_initial_parms: # 初期値の指定
            print(u'指定した初期値で学習を行います．')
            #各畳込み層のフィルタ数
            initial_num_filters = 191
            num_filters = trial.suggest_int('num_filters', initial_num_filters, initial_num_filters)
            
            #Pooling層の有無，0がFalse，1がTrue
            initial_Pooling_layer = 0
            Pooling_layer = trial.suggest_int('Pooling_layer', initial_Pooling_layer, initial_Pooling_layer)
            
            #Dense層の数
            num_layer = 5
            
            #Dense層のユニット数
            initial_mid_units = [426, 362, 383, 304, 405]
            mid_units = [int(trial.suggest_discrete_uniform('mid_units_'+str(i), initial_mid_units[i], initial_mid_units[i], 1)) for i in range(num_layer)]
            
            #活性化関数,leaky_relu
            initial_alpha = 0
            alpha = trial.suggest_uniform('alpha', initial_alpha, initial_alpha)
            activation = LeakyReLU(alpha)
            
            #optimizer,adam
            initial_learning_rate = 0.00065717
            learning_rate = trial.suggest_float("learning_rate", initial_learning_rate, initial_learning_rate)
            OPTIMIZER = Adam(learning_rate=learning_rate)
            
            #batch_size, 2^n, 2^10=1024
            initial_batch_size = 10
            batch_size_index = trial.suggest_int('batchsize', 9, 11)
        else:
            #各畳込み層のフィルタ数
            num_filters = trial.suggest_int('num_filters', 1, 256)
            
            #Pooling層の有無，0がFalse，1がTrue
            Pooling_layer = trial.suggest_int('Pooling_layer', 0, 1)
            
            #Dense層の数
            num_layer = 5
            
            #Dense層のユニット数
            mid_units = [int(trial.suggest_discrete_uniform('mid_units_'+str(i), 1, 500, 1)) for i in range(num_layer)]
            
            #活性化関数,leaky_relu
            alpha = trial.suggest_uniform('alpha', 0, 5.0e-01)
            activation = LeakyReLU(alpha)
            
            #optimizer,adam
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
            OPTIMIZER = Adam(learning_rate=learning_rate)
            
            #batch_size, 2^n, 2^10=1024
            batch_size_index = trial.suggest_int('batchsize', 9, 11)

        # Pooling層を使うか使わないか
        # if Pooling_layer == 0:
            # Pooling_switch = 'off'
        # elif Pooling_layer == 1:
            # Pooling_switch = 'on'

        # batch_size
        batch_size = 2**batch_size_index

        if model_type == 'Sequential':
            ''' Sequentialモデル '''
            print(u'Sequentialモデルによる学習を行います． outputの数：%d' % output_num)
            dense_list = [(Dense, mid_units[i], dict(activation=activation, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l2))) for i in num_layer]
            layers = (
                      (Conv2D, num_filters, dict(kernel_size=16, strides=8, padding='valid', activation=activation, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l2), input_shape=(32,32,2))),
#                      (DepthwiseConv2D, dict(kernel_size=16, strides=8, padding='valid', depth_multiplier=num_filters, activation=activation, depthwise_initializer='glorot_normal', input_shape=(32,32,2))),
                      (MaxPooling2D, dict(pool_size=(2, 2), strides=1, switch=Pooling_switch)),
                      (Flatten),
                      dense_list[0],
                      (Dropout, dropout_rate),
                      dense_list[1],
                      (Dropout, dropout_rate),
                      dense_list[2],
                      (Dropout, dropout_rate),
                      dense_list[3],
                      (Dropout, dropout_rate),
                      dense_list[4],
                      (Dropout, dropout_rate),
                      (Dense, output_num, dict(activation='linear', kernel_initializer='glorot_normal', name='dense_last', kernel_regularizer=regularizers.l2(l2))),
                      )
            with strategy.scope():
                model = my_sequential_model_builder(layers = layers, optimizer = OPTIMIZER, loss=['mae'], metrics='MAPE')
                for ily, layer in enumerate(model.layers):
                    # input layer
                    if ily == 0:
                        input = layer.input
                        h = input
                    # is dropout layer ?
                    if 'dropout' in layer.name:
                        # change dropout layer
                        h = Dropout(layer.rate)(h, training=True)
                    else:
                        h = layer(h)
            model.summary()

        elif model_type == 'functional_API':
            ''' functional APIモデル '''
            print(u'functional APIモデルによる学習を行います． outputの数：%d' % output_num)
            with strategy.scope():
                input1 = Input(shape = input_shape)
                input2 = Input(shape = input_shape)
                # 255で割るLambdaレイヤ
                normalize_layer = Lambda(lambda x: x/255.0) # change scale
                preprocessing_layer = Lambda(lambda x: tf.expand_dims(x, axis = -1))
                c2d = Conv2D(filters = num_filters, kernel_size = 16, strides=8, padding='valid', activation = activation, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l2))
                flatten_layer =  Flatten()
                # c2d.get_weights()[0]: weights, shape = (H, W, C, F)
                # c2d.get_weights()[1]: bias, shape = (F,), only in the case that use_bias is True
                x1 = flatten_layer(c2d(preprocessing_layer(normalize_layer(input1))))
                # x1 = flatten_layer.__call__(c2d.__call__(normalize_layer.__call__(input1))) でも良い
                # callメソッド -> https://qiita.com/ko-da-k/items/439d8cc3a0424c45214a
                x2 = flatten_layer(c2d(preprocessing_layer(normalize_layer(input2))))
                x = Concatenate()([x1, x2])
                for i in range(num_layer):
                    x = Dense(units = mid_units[i], activation = activation, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(l2))(x)
                output = Dense(units = output_num, activation='linear', kernel_initializer='glorot_normal', name='dense_last', kernel_regularizer=regularizers.l2(l2))(x)
                model = tf.keras.Model(inputs = [input1, input2], outputs = output)
                model.compile(optimizer = OPTIMIZER, loss = ['mae'], metrics = ['MAPE'])
            model.summary()
        
        #callbacksの設定
        #早期終了
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        #損失がnanになったら終了
        nan = keras.callbacks.TerminateOnNaN()
        
        #epoch毎にグラフ描画
        cb_figure = LossHistory(save_name = 'best_model_history', trial_num=trial._trial_id-1, metrics='my_mape', minimize_loss=minimize_loss, am_list=am_list, output_num=output_num, memmap_dir=memmap_dir)
        
        #畳み込み層の重みの共有
        copy_weights = CopyWeights(model)
        
        #モデルの保存
        save_checkpoint = keras.callbacks.ModelCheckpoint('weights{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1, save_freq=1)
        
        #使用するコールバックスの指定
        callbacks = [es, nan, cb_figure]

        # モデルの学習の設定
        verbose = 1
        epochs = 100
        
        # モデルの学習
        model_train(model, verbose, epochs, batch_size, callbacks, load_split_batch, model_type)
        
        #損失の比較，モデルの保存
        loss = history.history['val_loss'][-1]
        if minimize_loss > loss:
            minimize_loss = loss
            model.save(best_model_name)

        #メモリの開放
        keras.backend.clear_session()
        del model, history, OPTIMIZER
        gc.collect()
        
        #検証用データに対する損失が最小となるハイパーパラメータを求める
        return loss

def model_train(model, verbose, epochs, batch_size, callbacks, load_split_batch, model_type):
    if not load_split_batch:
        #訓練データを外から中に入れる
        x_train_copy = np.copy(x_train_data)
        y_train_copy = np.copy(y_train_data)
        x_val_copy = np.copy(x_val_data)
        y_val_copy = np.copy(y_val_data)
        print([x_train_copy[:,:,:,0].shape,x_train_copy[:,:,:,1].shape])
        
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


def restudy_by_monte_carlo_dropout():
    model = tf.keras.models.load_model(best_model_name, custom_objects={'LeakyReLU': LeakyReLU})

    for index, layer in enumerate(model.layers):
        if index == 0:
            input1 = layer.input
            h1 = input1
        elif index == 1:
            input2 = layer.input
            h2 = input2
        elif 'lambda' in layer.name or 'flatten' in layer.name:
            h1 = layer(h1)
            h2 = layer(h2)
        elif 'conv2d' in layer.name:
            c2d = layer
            h1 = c2d(h1)
            h2 = c2d(h2)
        elif 'concatenate' in layer.name:
            h = layer([h1, h2])
        elif 'dense' in layer.name and not 'last' in layer.name:
            h = layer(h)
            h = Dropout(dropout_rate)(h, training=True)
        elif 'last' in layer.name:
            h = layer(h)

    Model = tf.keras.Model([input1,input2],h)
    Model.compile(optimizer = model.optimizer, loss = 'mae', metrics = 'mape')
    Model.summary()

    for key, value in study.best_trial.params.items():
        if 'batch' in key:
            batch_size = 2 ** value
    # モデルの学習の設定
    verbose = 1
    epochs = 300

    #epoch毎にグラフ描画
    cb_figure = LossHistory(save_name = 'MC_model_history', am_list=am_list)
    callbacks = [cb_figure]

    model_train(Model, verbose, epochs, batch_size, callbacks, load_split_batch, model_type)
    model.save('MC_model.h5')

def input_str(message):
    return (raw_input if sys.version_info.major <= 2 else input)(message)

def input_int(message):
    while True:
        try:
            x = int((raw_input if sys.version_info.major <= 2 else input)(message))
            return x
        except:
            pass

def input_float(message):
    while True:
        try:
            x = float((raw_input if sys.version_info.major <= 2 else input)(message))
            return x
        except:
            pass

if __name__ == '__main__':
    interactive_mode = True
    data_determination = False
    use_memmap = False
    reset_optimize = False
    load_split_batch = False
    output_num = 2 # 出力数のデフォルト値
    output_axis = 0 # プレースホルダー
    monte_carlo_dropout = False
    dropout_rate = None # ドロップアウト層の値
    default_dropout_rate = 0.5 # デフォルトのドロップアウト層の値
    set_initial_parms = False
    n_jobs = 1 # シングルコアでの実行
    time_out = None
    check_param = False
    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        if sys.argv[i].lower().startswith('-h'):
            print(u'\n使い方:python %s -d dir_path' % os.path.basename(sys.argv[0]) + ' -d[irectory] dir_path ...\n' +
                   u'---- オプション ----\n' +
                   u'-d[irectory] dir_path         -> 訓練データがあるディレクトリを指定．zipファイルでも可．memmapファイルの入ったディレクトリでも可\n' +
                   u'-r[emove]                     -> 最適化の前回までのデータがある場合，それを消す．\n' +
                   u'-b[atch]                      -> 訓練データをバッチごとに読み込む．\n' +
                   u'-o[utput_num] number (axis)   -> 出力の数を指定，出力の数が1の場合は軸（x, y, z）も指定．\n' +
                   u'-m[onte]                      -> モンテカルロドロップアウト（予測時にもドロップアウト層を使うような学習モデルの作成）．\n' +
                   u'-i[nitial]                    -> 最適化の初期値を固定して実行．\n' +
                   u'-n n_jobs                     -> 並列処理の分割数をn_jobs個に設定．\n' +
                   u'-t[ime] time_out              -> [hour]最適化実行時間，time_out時間実行．\n' +
                   u'-c[onfirmation]               -> 最適なハイパーパラメータの確認．\n' +
                   u'-h[elp]                       -> ヘルプの表示．\n' +
                   u'\nデフォルトの設定\n' +
                   u'訓練データの読み込み設定  : すべて読み込む\n' +
                   u'前回までの最適化データ    : 前回の続きから開始する\n' +
                   u'並列計算における分割数   : シングルコアで実行\n' +
                   u'データの保存場所         : 作業フォルダ\n' +
                   u'出力数                 : 2\n' +
                   u'ドロップアウト層の値     : 0.5\n'
                   )
            quit()
        elif sys.argv[i].lower().startswith('-d'):
            i += 1
            data_directory = sys.argv[i]
            data_determination = True
        elif sys.argv[i].lower().startswith('-r'):
            reset_optimize = True
        elif sys.argv[i].lower().startswith('-b'):
            load_split_batch = True
        elif sys.argv[i].lower().startswith('-o'):
            i += 1
            output_num = int(sys.argv[i])
            if output_num == 1:
                i += 1
                output_axis = sys.argv[i]
        elif sys.argv[i].lower().startswith('-m'):
            monte_carlo_dropout = True
            dropout_rate = default_dropout_rate # ドロップアウト層の値
        elif sys.argv[i].lower().startswith('-i'):
            set_initial_parms = True
        elif sys.argv[i].lower().startswith('-n'):
            i += 1
            n_jobs = max(int(sys.argv[i]),1)
        elif sys.argv[i].lower().startswith('-t'):
            i += 1
            time_out = float(sys.argv[i])*60*60 # [s]時間を秒に変換
        elif sys.argv[i].lower().startswith('-c'):
            check_param = True
        i += 1

        
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    
    strategy = mode_detect() # ハードウェア情報取得
    study_name = 'optimize-CNN' # 最適化計算の名前
    study_name_path = study_name + '.db'
    y_dim = 3 # 変位の次元数
    memmap_dir = 'memmap' # デフォルトでmemmapを保存するディレクトリ
    first_divide = 0.2 # 8:2，学習データ:テストデータ
    second_devide = 0.25 # 6:2:2，訓練データ:検証データ:テストデータ
    model_type = 'functional_API' # Sequential / functional_API
    best_model_name = 'best_model.h5'
    exist_best_model = os.path.exists(best_model_name)# 解析ディレクトリに学習モデルがあるかの確認．
    input_shape = (32, 32) # 入力データの形状，32[pixel]×32[pixel]
    
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 

    # 予測時にもドロップアウト層を有効にした学習モデルで再学習を行う．
    if interactive_mode:
        if input_str('モンテカルロドロップアウトの学習モデルにしますか．（予測時にもドロップアウト層を使うような学習モデル．y/n）>> '.format(os.path.basename(sys.argv[0]))).lower().startswith('y'):
            monte_carlo_dropout = True
            interactive_mode = False
            while True:
                try:
                    dropout_rate = (raw_input if sys.version_info.major <= 2 else input)('ドロップアウト層の値を入力してください．(0~1．デフォルトは0.5) >> ')
                    if len(dropout_rate) == 0:
                        dropout_rate = default_dropout_rate
                        break
                    dropout_rate = float(dropout_rate)
                    assert 0.0 <= dropout_rate and dropout_rate <= 1.0
                    break
                except:
                    pass

    # モデル構造の確認
    if interactive_mode:
        check_model_type = input_str(u'モデル構造が%sになっています．このまま学習を開始しますか？（nで終了）>> ' % model_type)
        if check_model_type.lower().strip() == 'n':
            quit()
    
    # 出力数の決定
    if interactive_mode:
        output_num = input_int('出力の数を入力してください．>> ') # 扱うoutputの数
        if output_num == 1:
            output_axis = input_int('出力の軸（推定したい速度方向）を入力してください．（x or y or z） >> ')
    if output_num == 1: # 出力データのaxis = -2におけるインデックスを決定．x, y, z の順番に並んでいると考えている．
        if output_axis.lower().strip().startswith('x'):
            output_axis = 0
        elif output_axis.lower().strip().startswith('y'):
            output_axis = 1
        elif output_axis.lower().strip().startswith('z'):
            output_axis = 2
    
    
    # 学習データの読み込み
    if interactive_mode or not data_determination:
        data_directory = input_str('学習データのディレクトリを指定してください．>> ').strip()
    if any(['.npy' in i for i in os.listdir(data_directory)]): # 学習データのディレクトリの中身がmemmapの拡張子か判定．
        use_memmap = True
    
    # 学習データの読み込み条件の指定．
    if interactive_mode:
        if input_str('学習データをバッチごとに読み込みますか．（データが多いときはy，y/n）>> ').lower().strip().startswith('y'):
            load_split_batch = True
    
    # 前回までの最適化データの削除
    if interactive_mode:
        if input_str('前回までの最適化データがある場合，それを削除しますか．（y/n）>> ').lower().startswith('y'):
            reset_optimize = True
    if reset_optimize and not check_param:
        if os.path.exists(study_name_path):
            os.remove(study_name_path)
    
    # Optunaの設定，並列処理数の指定
    if interactive_mode:
        if input_str('Optunaによる最適化に指定した初期値を使いますか．（初期値は {} に直接書き込む必要があります．y/n）>> '.format(os.path.basename(sys.argv[0]))).lower().startswith('y'):
            set_initial_parms = True
        n_jobs = max(input_int('memmapファイル作成時の並列処理の分割数を指定してください．（最大分割数{}，1だとシングルコアで計算．）>> '.format(multiprocessing.cpu_count())), 1)
        time_out = input_str('Optunaによる最適化の計算時間を指定してください．（単位[hour]，指定しなければずっと計算を行う．）>> ')
        try:
            time_out=float(time_out)*60*60 # hour -> secondに変換
        except:
            time_out=None

    # 最適化関数のインスタンス化
    study = optuna.create_study(storage='sqlite:///optimize-CNN.db',
                                study_name=study_name,
                                load_if_exists=True,
                                pruner=optuna.pruners.PercentilePruner(60, interval_steps=10), # 枝刈りの設定，詳しくはhttps://optuna.readthedocs.io/en/stable/reference/pruners.html
                                )
    try:
        minimize_loss = study.best_trial.value
    except:
        minimize_loss = 1.0e5 # 最小値問題なので，初期値は何でもいいので大きい値．

    if not check_param:
        if not use_memmap: # memmapファイルを使わない場合
            os.makedirs(memmap_dir, exist_ok=True) # memmapファイルを保存するディレクトリの作成．
            if data_directory.endswith(u'.zip'):
                with zipfile.ZipFile(data_directory) as zf:
                    zf.extractall(os.path.dirname(data_directory))
                    data_directory = os.path.splitext(os.path.basename(data_directory))[0]
            
            # データを使いやすい形に整理
            if not load_split_batch:
                dataset = MyDataset(y_dim)
                input_data = dataset.im2array(data_directory, n_jobs=n_jobs) # 入力データの画像 -> 配列に変換
                
                 # 出力データの読み込み
                output_path = glob(os.path.join(data_directory,'**', 'dataset_output.txt'), recursive=True)[0]
                output_data = get_input_output_from_file(output_path, top_skip = 1, input_columns = None, output_columns = range(y_dim), delimiter = None, encoding = 'UTF-8')
                
                print(u'総データ数： ' + str(output_data.shape[0]))
                
                # 学習用データとテストデータに分割
                x_learn, x_test_data, y_learn, y_test_data = train_test_split(input_data, output_data, test_size=first_divide)
                # 学習用データを訓練データと検証データに分割
                x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(x_learn, y_learn, test_size=second_devide)

                # 入力データの正規化，最大輝度数で割っている．
                max_luminance = 255 # 最大輝度数
                x_train_data = x_train_data/max_luminance
                x_val_data = x_val_data/max_luminance
                x_test_data = x_test_data/max_luminance
                
                dataset.save_memmap((x_train_data, x_test_data, x_val_data, y_val_data, y_train_data, y_test_data), memmap_dir) # memmapファイルを保存
                
                # メモリの開放
                del input_data, output_data, x_test_data, y_test_data
                gc.collect()
                
                x_train_data = np.memmap(filename=os.path.join(memmap_dir, 'x_train_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
                y_train_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_train_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
                x_val_data = np.memmap(filename=os.path.join(memmap_dir, 'x_val_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
                y_val_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_val_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)

            elif load_split_batch:
                print('訓練データをバッチごとに読み込みます．')
                # 訓練データをバッチごとに読み込むときのデータセット
                dataset = MyGeneratorDataset(y_dim=y_dim, n_jobs=n_jobs)
                data_set = dataset.get_paths(data_directory)
                print(u'総データ数： ' + str(len(data_set)))
                learning_data, test_data = train_test_split(data_set, test_size=first_divide) # データセットを学習用データとテストデータに分割
                train_data, val_data = train_test_split(learning_data, test_size=second_devide) # 学習用データを訓練データと検証データに分割
                data_size = dataset.generate_memmap((train_data, val_data, test_data), memmap_dir)
        
        elif use_memmap:
            memmap_dir = data_directory
            if not load_split_batch:
                x_train_data = np.memmap(filename=os.path.join(memmap_dir, 'x_train_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
                y_train_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_train_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
                x_val_data = np.memmap(filename=os.path.join(memmap_dir, 'x_val_data.npy'), dtype=np.float32, mode='r').reshape(-1, 32, 32, 2)
                y_val_data = read_ymemmap(filename=os.path.join(memmap_dir, 'y_val_data.npy'), y_dim=y_dim, output_num=output_num, output_axis=output_axis)
            elif load_split_batch:
                data_size = []
                for i in ['train_data', 'val_data', 'test_data']:
                    y_path = 'y_' + i + '.npy'
                    y_memmap = np.memmap(filename=os.path.join(memmap_dir, y_path), dtype=np.float32, mode='r')
                    data_size.append(y_memmap.reshape(-1, y_dim).shape[0]) # memmapファイルを利用して学習を行う．
                    del y_memmap
        am_list = calc_am(memmap_dir, y_dim, output_num, output_axis)
        if monte_carlo_dropout and exist_best_model:
            restudy_by_monte_carlo_dropout()
        else:
            objective = Objective(strategy, input_shape, y_dim, output_num, output_axis, memmap_dir, minimize_loss, load_split_batch, am_list, dropout_rate, set_initial_parms, model_type, best_model_name)
            study.optimize(objective, timeout=time_out)
        if monte_carlo_dropout and not exist_best_model:
            restudy_by_monte_carlo_dropout()

    
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print()
    trial = study.best_trial
    print("Best trial: " + str(trial._trial_id - 1))
    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
