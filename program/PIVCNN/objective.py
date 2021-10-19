#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:26:23
# objective.py

import tensorflow as tf
import tensorflow.keras as keras
import gc

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
from tensorflow.keras import callbacks
from tensorflow.keras import backend

def objective():
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