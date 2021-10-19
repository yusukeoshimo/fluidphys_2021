#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:26:23
# monte_carlo_dropout.py

import tensorflow as tf

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU

from my_model import model_train
from my_callbacks import LossHistory

def restudy_by_monte_carlo_dropout(best_model_name, am_list, load_split_batch, model_type, memmap_dir, y_dim, output_num, output_axis, data_size, dropout_rate, study):
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
    epochs = 100

    #epoch毎にグラフ描画
    cb_figure = LossHistory(save_name = 'MC_model_history', memmap_dir=memmap_dir, am_list=am_list, output_num=output_num)
    callbacks = [cb_figure]

    Model, history = model_train(Model, verbose, epochs, batch_size, callbacks, load_split_batch, model_type, memmap_dir, y_dim, output_num, output_axis, data_size)
    Model.save('MC_model.h5')