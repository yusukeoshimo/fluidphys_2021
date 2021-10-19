#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:44:31
# data_processing.py

import numpy as np
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from joblib import Parallel, delayed

from read_data import read_image, get_input_output_from_file

class MyDataset:
    def __init__(self, y_dim=None, global_dict=None):
        self.y_dim = y_dim
        self.global_dict = global_dict

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
            for key, value in self.global_dict:
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
    def __init__(self, y_dim=None, n_jobs=1, global_dict=None):
        self.y_dim = y_dim
        self.n_jobs = n_jobs
        self.global_dict = global_dict

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
            for key, value in self.global_dict:
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