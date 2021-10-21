#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import cv2
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import sys

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

        white_noise = random.uniform(0,255*0.01)
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

    def post_processing(self):
        stick_output_list = [os.path.join(self.dir_name_0,'dataset_output_{}.txt'.format(i)) for i in range(self.logical_processor)]
        with open(os.path.join(self.dir_name_0,'dataset_output.txt'),'w') as f:
            f.write('# ux   uy   uz \n')
        for data_output_path in tqdm(stick_output_list):
            with open(os.path.join(self.dir_name_0,'dataset_output.txt'),'a') as f:
                np.savetxt(f,np.loadtxt(data_output_path))

class Data2Memmap():
    def __init__(self,dir_name_0,dir_name_1,dataset_num_per_process,logical_processor,size):
        self.dir_name_0 = dir_name_0
        self.dir_name_1 = dir_name_1
        self.dataset_num_per_process = dataset_num_per_process
        self.logical_processor = logical_processor
        self.size = size

        
    # ディレクトリの準備
    def pre_processing(self):
        if os.path.exists(self.dir_name_1):
            shutil.rmtree(self.dir_name_1)
        os.mkdir(self.dir_name_1)

    def data2memmap(self):
        # y2memmap
        y_data = np.loadtxt(os.path.join(self.dir_name_0,'dataset_output.txt'))
        fp = np.memmap(os.path.join('memmap','y_test.npy'), dtype='float32', mode='w+', shape=y_data.shape)
        del fp
        fp = np.memmap(os.path.join('memmap','y_test.npy'), dtype='float32', mode='r+', shape=y_data.shape)
        fp[:] = y_data
        del fp
        del y_data

        # x2memmap
        fp = np.memmap(os.path.join('memmap','x_test.npy'), dtype='float32', mode='w+', shape=(self.logical_processor*self.dataset_num_per_process,self.size,self.size,2))
        del fp
        for processor in tqdm(range(self.logical_processor)):
            for image_num in range(self.dataset_num_per_process):
                origin_arr = cv2.imread(os.path.join(self.dir_name_0,str(processor),'origin_{}_{}.png'.format(processor,image_num)),0)
                next_arr = cv2.imread(os.path.join(self.dir_name_0,str(processor),'next_{}_{}.png'.format(processor,image_num)),0)
                arr = np.vstack((origin_arr,next_arr)).reshape(1,2,self.size,self.size).transpose(0,2,3,1)/255
                fp = np.memmap(os.path.join('memmap','x_test.npy'), dtype='float32', mode='r+', shape=(self.logical_processor*self.dataset_num_per_process,self.size,self.size,2))
                fp[processor*self.dataset_num_per_process+image_num:processor*self.dataset_num_per_process+image_num+1,:,:,:] = arr


if __name__ =='__main__':
    print('このプログラムは作業ディレクトリ内のresultフォルダを削除し新たなデータを作成します')
    cwd_path = input('作業ディレクトリを入力してください>')
    
    # MkImageクラスのパラメータ*********************************************************************************************************
    logical_processor = int(input('何コアで並列計算させますか？上限は{}です>'.format(multiprocessing.cpu_count())))
    hope_dataset_num = int(input('データセット数>'))
    dir_name_0 = 'result' # cwd直下のデータを保存するディレクトリ
    eq_coef = 240 # 輝度計算の係数
    sigma_l = 5 # レーザーシートの厚み [pixel]
    size = 32 # 画像のサイズ [pixel]
    times = 2 # 計算領域のxy平面の一辺のサイズ / 画像の一辺サイズ
    depth = 32 # 計算領域の深さ [pixel]
    particle_num_min = 70 # 粒子数の最小値 [個]
    particle_num_max = 350 # 粒子数の最大値 [個]
    d_p_min = 2.4 # 粒子径の最小値 [個]
    d_p_max = 2.6 # 粒子径の最大値 [個]
    graph_bool = False # 速度場のグラフを描写するか否か
    # *********************************************************************************************************************************

    # Data2Memmapクラスのパラメータ******************************************************************************************************
    dir_name_1 = 'memmap' # cwd直下のデータを保存するディレクトリ
    # *********************************************************************************************************************************

    # 作業ディレクトリにチェンジディレクトリ
    os.chdir(cwd_path)
    
    # データセットを並列計算によって作成
    mkimage = MkImage(logical_processor=logical_processor,
                      hope_dataset_num=hope_dataset_num,
                      dir_name_0=dir_name_0,
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
    mkimage.remove_dir()
    mkimage.mk_dir()
    processor = [processor for processor in range(logical_processor)]
    p = Pool(logical_processor)
    p.map(mkimage.main,processor)

    # 並列計算の結果をまとめる
    mkimage.post_processing()

    # データセットをnpyファイルに変換
    data2memmap = Data2Memmap(dir_name_0=dir_name_0,
                              dir_name_1=dir_name_1,
                              dataset_num_per_process=mkimage.dataset_num_per_process,
                              logical_processor=logical_processor,
                              size=size)
    data2memmap.pre_processing()
    data2memmap.data2memmap()