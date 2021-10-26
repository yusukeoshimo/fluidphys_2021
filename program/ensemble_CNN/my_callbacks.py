#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-09-29 15:49:06
# my_callbacks.py

import os
import numpy as np
from adjustText import adjust_text
import matplotlib

#バックエンドを指定
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tensorflow.keras import callbacks


#動的グラフを描画するためのクラス，ネット参照
class LossHistory(callbacks.Callback):
    def __init__(self, save_name=None, save_dir=None, model_num=0, output_num=None, am_list=None):
        # コンストラクタに保持用の配列を宣言しておく
        self.save_name = save_name
        self.save_dir = save_dir
        self.model_num = model_num
        self.loss = 'loss'
        if output_num == 2:
            self.ylabel = 'loss/$1/N\sum_{k=1}^{N} (|x_k|+|y_k|)/$' + str(output_num)
        elif output_num == 3:
            self.ylabel = 'loss/$1/N\sum_{k=1}^{N} (|x_k|+|y_k|+|z_k|)/$' + str(output_num)
        try:
            if am_list is not None:
                self.train_am = am_list['train_am']
                self.val_am = am_list['val_am']
                self.test_am = am_list['test_am']
        except:
            pass
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
        # loss_history.txtにデータを保存
        array_epoch = np.arange(len(self.train_loss)).reshape(-1, 1)
        array_train_loss = np.array(self.train_loss).reshape(-1, 1)
        array_val_loss = np.array(self.val_loss).reshape(-1, 1)
        array_train_metrics = np.array(self.train_metrics).reshape(-1, 1)
        array_val_metrics = np.array(self.val_metrics).reshape(-1, 1)
        loss_history = np.hstack((array_epoch, array_train_loss, array_val_loss, array_train_metrics, array_val_metrics))
        with open(os.path.join(self.save_dir, 'model_{}_loss_history.txt'.format(self.model_num)), 'w') as f:
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
        plt.title('learning curve ensemble model {}: '+str(self.model_num))
        plt.xlabel('epoch')
        plt.ylabel(self.ylabel)
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
        
        fig.savefig(os.path.join(self.save_dir, self.save_name + '.png'), bbox_inches="tight", pad_inches=0.05)
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
