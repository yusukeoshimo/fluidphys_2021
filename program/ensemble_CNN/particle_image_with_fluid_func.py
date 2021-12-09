#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-25 20:36:45
# particle_image_with_fluid_func.py

import shutil
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from glob import glob
from natsort import natsorted

from convenient import remake_dir
from read_data import read_image, get_input_output_from_file

class MkImage():
    def __init__(self,
                 logical_processor,
                 hope_dataset_num,
                 dir_name_0,
                 eq_coef,
                 sigma_l,
                 size,
                 times,
                 depth,
                 particle_num_min,
                 particle_num_max,
                 d_p_min,
                 d_p_max,
                 graph_bool):
        
        self.logical_processor = logical_processor # 並列計算のコア数
        self.dataset_num_per_process = hope_dataset_num//self.logical_processor # 1プロセスで作るデータ数
        self.dir_name_0 = dir_name_0 # cwd直下のデータを保存するディレクトリ
        self.eq_coef = eq_coef # 輝度計算の係数
        self.sigma_l = sigma_l # レーザーシートの厚み [pixel]
        self.size = size # 画像のサイズ [pixel]
        self.times = times # 計算領域のxy平面のサイズ / 画像サイズ
        self.depth = depth # 計算領域の深さ [pixel]
        self.width = self.size*self.times # 計算領域の幅 [pexel]
        self.height = self.size*self.times # 計算領域の高さ [pexel]
        self.particle_num_min = particle_num_min # 粒子数の最小値 [個]
        self.particle_num_max = particle_num_max # 粒子数の最大値 [個]
        self.d_p_min = d_p_min # 粒子径の最小値 [個]
        self.d_p_max = d_p_max # 粒子径の最大値 [個]
        self.x_center = self.width/2 # 計算領域のxの中心座標 [pixel]
        self.y_center = self.height/2 # 計算領域のyの中心座標 [pixel]
        self.z_center = self.depth/2 # 計算領域のzの中心座標 [pixel]
        self.graph_bool = graph_bool # 速度場のグラフを描写するか否か

    def remove_dir(self):
        if os.path.exists(self.dir_name_0):
            shutil.rmtree(self.dir_name_0)
    
    def mk_dir(self):
        if not os.path.exists(self.dir_name_0):
            os.mkdir(self.dir_name_0)
        
    def generate_particle(self):
        particle_num = random.randint(self.particle_num_min,self.particle_num_max)
        x_p = np.random.uniform(0,self.width-1,particle_num).reshape(-1,1,1)
        y_p = np.random.uniform(0,self.height-1,particle_num).reshape(-1,1,1)
        z_p = np.random.uniform(0,self.depth-1,particle_num).reshape(-1,1,1)
        d_p = np.random.uniform(self.d_p_min,self.d_p_max,particle_num).reshape(-1,1,1)
        return x_p,y_p,z_p,d_p

    def generate_flow(self,x,y,z,random_bool):
        if random_bool:
            self.cx1 = random.uniform(-0.01,0.01)
            self.cx2 = random.uniform(-0.01,0.01)
            self.cx3 = random.choice([-1,1])*random.uniform(8.1,8.1)
            self.cy1 = random.uniform(-0.01,0.01)
            self.cy2 = random.uniform(-0.01,0.01)
            self.cy3 = random.choice([-1,1])*random.uniform(8.1,8.1)
            self.cz1 = random.uniform(-0.0015,0.0015)
            self.cz2 = random.uniform(-0.0015,0.015)
            self.cz3 = random.choice([-1,1])*random.uniform(1.4,1.4)
            self.cxy = random.uniform(-0.01,0.01)
            self.cxz = random.uniform(-0.0015,0.0015)
            self.cyz = random.uniform(-0.0015,0.0015)
            self.theta = math.radians(random.uniform(-180,180))
            self.u_random = random.uniform(0,8)
        
        # 流れ場を作る
        ux = self.cx1*z+self.cx2*y-self.cxz*x+self.cxy*x+self.cx3
        uy = self.cy1*z+self.cyz*y-self.cxy*y+self.cy2*x+self.cy3
        uz = -self.cyz*z+self.cxz*z+self.cz1*y+self.cz2*x+self.cz3

        # 流れ場をランダムに回転させる
        rotated_ux = ux*math.cos(self.theta)-uy*math.sin(self.theta)
        rotated_uy = uy*math.cos(self.theta)+ux*math.sin(self.theta)

        # 流れ場を任意の範囲内の大きさにする
        uxy_abs = np.sqrt(((self.cx1*self.z_center+self.cx2*self.y_center-self.cxz*self.x_center+self.cxy*self.x_center+self.cx3)**2+
                            (self.cy1*self.z_center+self.cyz*self.y_center-self.cxy*self.y_center+self.cy2*self.x_center+self.cy3)**2))
        ux = (rotated_ux/uxy_abs)*self.u_random
        uy = (rotated_uy/uxy_abs)*self.u_random
        uz = uz/uxy_abs*self.u_random

        return ux,uy,uz

    def mk_image(self,x_p,y_p,z_p,d_p):
        x = np.arange(self.width)
        y = np.arange(self.height).reshape(-1,1)

        max_luminance = 255 # 最大輝度
        white_noise = random.uniform(0, max_luminance*0.01)
        luminance_array = np.sum(self.eq_coef*np.exp(-(z_p - self.depth/2)**2/self.sigma_l**2)*np.exp(-((x-x_p)**2+(y-y_p)**2)/(d_p/2)**2),axis=0)+white_noise # 輝度の計算
        image = luminance_array[self.size*(self.times - 1)//2:self.size*(self.times + 1)//2, self.size*(self.times - 1)//2:self.size*(self.times + 1)//2]
        return image
    
    def flow_visualization(self,x,y,graph_ux,graph_uy,processor,dataset_num):
        plt.figure()
        plt.quiver(x,y,graph_ux,graph_uy,color='red',angles='xy',scale_units='xy', scale=1)
        plt.xlim([0,self.size])
        plt.ylim([0,self.size])
        plt.xticks([0,self.size/2,self.size])
        plt.yticks([0,self.size/2,self.size])
        plt.grid()
        plt.draw()
        fig_path = os.path.join(self.dir_name_0,str(processor),'fig_{}_{}'.format(processor, dataset_num))
        plt.savefig(fig_path)

    def main(self,processor):
        for dataset_num in tqdm(range(self.dataset_num_per_process)):

            # 粒子をランダムに配置
            x_p, y_p, z_p, d_p = self.generate_particle()

            # 粒子画像を作成 & 保存
            image_1 = self.mk_image(x_p, y_p, z_p, d_p)
            if dataset_num == 0:
                os.mkdir(os.path.join(self.dir_name_0,str(processor)))
            cv2.imwrite(os.path.join(self.dir_name_0,str(processor),'origin_{}_{}.png'.format(processor,dataset_num)),image_1)

            # 粒子の移動を計算
            ux,uy,uz = self.generate_flow(x_p,y_p,z_p,True)

            # 移動後の粒子画像を作成 & 保存
            image_2 = self.mk_image(x_p+ux,y_p+uy,z_p+uz,d_p)
            cv2.imwrite(os.path.join(self.dir_name_0,str(processor),'next_{}_{}.png'.format(processor,dataset_num)),image_2)

            # 画像中心の速度を計算 & 保存
            ux_center,uy_center,uz_center = self.generate_flow(self.x_center,self.y_center,self.z_center,False)
            data_output_path = os.path.join(self.dir_name_0,'dataset_output_{}.txt'.format(processor))
            if dataset_num == 0:
                with open (data_output_path,'w') as f:
                    f.write('# ux   uy   uz \n')
            with open(data_output_path,'a') as f:
                # print(np.array([ux_center,uy_center,uz_center]))
                np.savetxt(f,np.array([ux_center,uy_center,uz_center]).reshape(1,-1))
            
            # 速度場を描写　＆　保存
            if self.graph_bool:
                gridwidth = self.size/6
                x, y = np.meshgrid(np.arange(0, self.width, gridwidth), np.arange(0, self.height, gridwidth))
                z = self.z_center
                graph_ux, graph_uy , graph_uz = self.generate_flow(x,y,z,False)
                self.flow_visualization(x, y, graph_ux, graph_uy, processor, dataset_num)

    def post_processing(self, y_dim):
        stick_output_list = [os.path.join(self.dir_name_0,'dataset_output_{}.txt'.format(i)) for i in range(self.logical_processor)]
        with open(os.path.join(self.dir_name_0,'dataset_output.txt'),'w') as f:
            f.write('# ux   uy   uz \n')
        for data_output_path in tqdm(stick_output_list):
            with open(os.path.join(self.dir_name_0,'dataset_output.txt'),'a') as f:
                np.savetxt(f,np.loadtxt(data_output_path).reshape(-1, y_dim))

# memmapファイルを作るクラス
class Data2Memmap:
    def __init__(self, y_dim=None, n_jobs=1):
        self.y_dim = y_dim
        self.n_jobs = n_jobs

    def get_paths(self, dir_name):
        self.dir_name = dir_name # 学習データ（画像）の入ったディレクトリを指定
        all_input_paths = []

        dir_list = natsorted([f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))])
        for i in tqdm(dir_list, desc='訓練データ読み込み 進捗'):
            
            file_count = sum(os.path.isfile(os.path.join(dir_name, i, name)) for name in os.listdir(os.path.join(dir_name, i)))
            input_paths = Parallel(n_jobs=self.n_jobs, backend='threading')([delayed(self.append_input_path)(i, j) for j in range(file_count//2)])
            all_input_paths.extend(input_paths)
            del input_paths
        output_path = glob(os.path.join(dir_name,'**', 'dataset_output.txt'), recursive=True)[0]
        output_data = get_input_output_from_file(output_path, skip_word = '#', input_columns = None, output_columns = range(self.y_dim), delimiter = None, encoding = 'UTF-8')
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

    def generate_memmap(self, data, memmap_dir, variable_dict):
        if not isinstance(data, tuple):
            data = (data,)
        size_list = []
        for i in tqdm(data, desc='memmap作成 全体の進捗'):
            size_list.append(len(i))
            for key, value in variable_dict: # グローバル／ローカル変数をfor分でループさせる
                if id(i) == id(value): # 同じid値のときにその変数名を file_name に入れる
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
        # X_memmap.shape = (データ数，行数, 列数, 深さ)
        # example X
        # 画像1の形状:array_shape = (行数, 列数)
        #            <--------- 列 ---------->
        #        ^   [[ 0  1  0 ...  0  0  0]
        #        |    [ 6 10  4 ...  0  0  0]
        #        |    [30 46 20 ...  0  0  1]
        #        行    ...
        #        |    [ 0  0  0 ...  0  0  0]
        #        |    [ 0  0  0 ...  0  0  0]
        #        v    [ 0  0  0 ...  0  0  0]]
        # 画像2の形状:[[ 0  0  0 ... 33  8  1]
        #             [ 0  0  0 ... 55 13  1]
        #             [ 0  0  0 ... 28  7  0]
        #             ...
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]
        #             [ 0  0  0 ...  0  0  0]]
        # 配列の結合:array_shape = (行数, 列数, 深さ)
        #           [[[ 0  0]
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
        
        # yのデータをmemmapのファイルへ書き込む。
        y_memmap[j] = np.array(y)
        # Y:[v_xk, v_yk]
        # example y_memmap
        # y_memmap.shape = (データ数，速度の次元数)
        #            <--------- 速度の次元数 --------->
        #        ^   [[     v_x0,     v_y0,     v_z0]
        #        |    [     v_x1,     v_y1,     v_z1]
        #        |    [     v_x2,     v_y2,     v_z2]
        #     データ数  ...
        #        |    [ v_x(k-2), v_y(k-2), v_z(k-2)]
        #        |    [ v_x(k-1), v_y(k-1), v_z(k-1)]
        #        v    [     v_xk,     v_yk,     v_zk]]

        del X_memmap, y_memmap

def mkdata(data_num, memmap_dir, y_dim, split_rate=None):
    # MkImageクラスのパラメータ*********************************************************************************************************
    logical_processor = multiprocessing.cpu_count() - 1
    if logical_processor >= data_num:
        logical_processor = data_num
    hope_dataset_num = data_num
    data_directory = 'result_' + str(data_num) # cwd直下のデータを保存するディレクトリ
    eq_coef = 240 # 輝度計算の係数
    sigma_l = 10 # レーザーシートの厚み [pixel]
    size = 32 # 画像のサイズ [pixel]
    times = 2 # 計算領域のxy平面の一辺のサイズ / 画像の一辺サイズ
    depth = 32 # 計算領域の深さ [pixel]
    particle_num_min = 70 # 粒子数の最小値 [個]
    particle_num_max = 350 # 粒子数の最大値 [個]
    d_p_min = 2.4 # 粒子径の最小値 [個]
    d_p_max = 2.6 # 粒子径の最大値 [個]
    graph_bool = False # 速度場のグラフを描写するか否か
    # *********************************************************************************************************************************

    # 前回のデータがある場合，削除
    remake_dir(data_directory)
    remake_dir(memmap_dir)

    # データセットを並列計算によって作成
    mkimage = MkImage(logical_processor=logical_processor,
                      hope_dataset_num=hope_dataset_num,
                      dir_name_0=data_directory,
                      eq_coef=eq_coef,
                      sigma_l=sigma_l,
                      size=size,
                      times=times,
                      depth=depth,
                      particle_num_min=particle_num_min,
                      particle_num_max=particle_num_max,
                      d_p_min=d_p_min,
                      d_p_max=d_p_max,
                      graph_bool=graph_bool)
    processor = [processor for processor in range(logical_processor)]
    p = Pool(logical_processor)
    p.map(mkimage.main,processor)

    # 並列計算の結果をまとめる
    mkimage.post_processing(y_dim)

    dataset = Data2Memmap(y_dim=y_dim, n_jobs=logical_processor)
    data_set = dataset.get_paths(data_directory) # 画像のパスをリストで返す
    if split_rate is not None:
        if hasattr(split_rate,'__iter__'):
            if len(split_rate) == 1:
                # 全データを訓練データと検証データに分割
                train_data, val_data = train_test_split(data_set, test_size=split_rate[0])
                data_list = (train_data, val_data)
            elif len(split_rate) == 2:
                # データセットを学習用データとテストデータに分割
                learning_data, test_data = train_test_split(data_set, test_size=split_rate[0])
                # 学習用データを訓練データと検証データに分割
                train_data, val_data = train_test_split(learning_data, test_size=split_rate[1])
                data_list = (train_data, val_data, test_data)
            else:
                print('len(split_rate) != 1, 2 \n' +
                      '学習／テストデータの分割ができませんでした．')
                sys.exit(1) # 異常終了
        else:
            # 全データを訓練データと検証データに分割
            train_data, val_data = train_test_split(data_set, test_size=split_rate)
            data_list = (train_data, val_data)
    else:
        test_data = data_set
        data_list = (test_data,)
    # memmapファイルの作成
    data_size = dataset.generate_memmap(data_list, memmap_dir, locals().items())
    # data_size = [len(data[0]), len(data[1]), ...]

    return data_size

if __name__ =='__main__':
    # 後のif分判定に使うフラグ
    interactive_mode = True
    # プレースホルダー
    data_num = None

    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        if sys.argv[i].lower().startswith('-h'): # ヘルプの表示
            print(u'使い方: python {}'.format(os.path.basename(sys.argv[0])) +
                  u' -d[ata_number] data_num .... \n' +
                  u'---- オプション ----\n' +
                  u'-d[ata_number] data_num     -> 学習またはテストに使うデータ数を data_num で指定．\n' +
                  u'-h[elp]                     -> ヘルプの表示．\n'
                  )
            sys.exit(0) # 正常終了, https://www.sejuku.net/blog/24331
        elif sys.argv[i].lower().startswith('-d'): # データ数の指定
            i += 1
            data_num = int(sys.argv[i])
        i += 1

    # 学習，推論に使用するデータ数が指定されているのか確認，学習／推論する際に必要
    if data_num is None:
        from convenient import input_int
        data_num = input_int('学習／推論に使うデータ数を指定してください．>> ')
        assert data_num != '' # 何も入力していない場合エラー

    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    y_dim = 3 # 変位の次元数
    memmap_dir = 'memmap_' + str(data_num) # デフォルトでmemmapを保存するディレクトリ
    split_rate = [0.2, 0.25]
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 

    mkdata(data_num, memmap_dir, y_dim, split_rate)
