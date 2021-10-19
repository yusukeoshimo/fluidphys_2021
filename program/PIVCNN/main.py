#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:26:23
# main.py

#---- オプション ----
#-d[irectory] dir_path         -> 訓練データがあるディレクトリを指定．zipファイルでも可．memmapファイルの入ったディレクトリでも可
#-r[emove]                     -> 最適化の前回までのデータがある場合，それを消す．
#-b[atch]                      -> 訓練データをバッチごとに読み込む．
#-o[utput_num] number (axis)   -> 出力の数を指定，出力の数が1の場合は軸（x, y, z）も指定．
#-i[nitial]                    -> 最適化の初期値を固定して実行．
#-m[onte]                      -> モンテカルロドロップアウト（予測時にもドロップアウト層を使うような学習モデルの作成）．
#-n n_jobs                     -> 並列処理の分割数をn_jobs個に設定．
#-tr[ial] n_trials             -> 最適化試行回数，n_trials回実行．．
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
import os
import numpy as np
from glob import glob
import gc
import optuna
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import zipfile
import signal

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

from read_data import get_input_output_from_file, read_ymemmap
from data_processing import MyDataset, MyGeneratorDataset
from my_model import model_train, calc_am
from my_callbacks import LossHistory, CopyWeights
from monte_carlo_dropout import restudy_by_monte_carlo_dropout
from convenient import input_float, input_int, input_str

def objective(trial):
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

    print(u'functional APIモデルによる学習を行います． outputの数：%d' % output_num)
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
    cb_figure = LossHistory(save_name = 'best_model_history', memmap_dir=memmap_dir, trial_num=trial._trial_id-1, output_num=output_num, am_list=am_list)
    
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
    model, history = model_train(model, verbose, epochs, batch_size, callbacks, load_split_batch, memmap_dir, y_dim, output_num, output_axis, data_size)
    
    #損失の比較，モデルの保存
    loss = history.history['val_loss'][-1]
    global minimize_loss
    if minimize_loss > loss:
        minimize_loss = loss
        model.save(best_model_name)

    #メモリの開放
    keras.backend.clear_session()
    del model, history, OPTIMIZER
    gc.collect()
    
    #検証用データに対する損失が最小となるハイパーパラメータを求める
    return loss

if __name__ == '__main__':
    interactive_mode = True
    data_determination = False
    use_memmap = False
    reset_optimize = False
    load_split_batch = False
    batch_determination = False
    monte_carlo_dropout = False
    dropout_rate = None # ドロップアウト層の値
    default_dropout_rate = 0.5 # デフォルトのドロップアウト層の値
    pre_training_MC = False
    output_num = 2 # 出力数のデフォルト値
    output_axis = 0 # プレースホルダー
    set_initial_parms = False
    n_jobs = 1 # シングルコアでの実行
    time_out = None
    check_param = False
    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        batch_determination = True
        if sys.argv[i].lower().startswith('-h'):
            print(u'\n使い方:python %s -d dir_path' % os.path.basename(sys.argv[0]) + ' -d[irectory] dir_path ...\n' +
                   u'---- オプション ----\n' +
                   u'-d[irectory] dir_path         -> 訓練データがあるディレクトリを指定．zipファイルでも可．memmapファイルの入ったディレクトリでも可\n' +
                   u'-r[emove]                     -> 最適化の前回までのデータがある場合，それを消す．\n' +
                   u'-b[atch]                      -> 訓練データをバッチごとに読み込む．\n' +
                   u'-m[onte]                      -> モンテカルロドロップアウト（予測時にもドロップアウト層を使うような学習モデルの作成）．\n' +
                   u'-om[optuna monte]             -> モンテカルロドロップアウトによる学習の前にOptunaによる最適化を行う．\n' +
                   u'-o[utput_num] number (axis)   -> 出力の数を指定，出力の数が1の場合は軸（x, y, z）も指定．\n' +
                   u'-i[nitial]                    -> 最適化の初期値を固定して実行．\n' +
                   u'-n n_jobs                     -> 並列処理の分割数をn_jobs個に設定．\n' +
                   u'-tr[ial] n_trials             -> 最適化試行回数，n_trials回実行．\n' +
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
        elif sys.argv[i].lower().startswith('-m'):
            monte_carlo_dropout = True
            dropout_rate = default_dropout_rate # ドロップアウト層の値
        elif sys.argv[i].lower().startswith('-om'):
            pre_training_MC = True
            dropout_rate = default_dropout_rate # ドロップアウト層の値
        elif sys.argv[i].lower().startswith('-o'):
            i += 1
            output_num = int(sys.argv[i])
            if output_num == 1:
                i += 1
                output_axis = sys.argv[i]
        elif sys.argv[i].lower().startswith('-i'):
            set_initial_parms = True
        elif sys.argv[i].lower().startswith('-n'):
            i += 1
            n_jobs = max(int(sys.argv[i]),1)
        elif sys.argv[i].lower().startswith('-tr'):
            i += 1
            n_trials = int(sys.argv[i])
        elif sys.argv[i].lower().startswith('-t'):
            i += 1
            time_out = float(sys.argv[i])*60*60 # [s]時間を秒に変換
        elif sys.argv[i].lower().startswith('-c'):
            check_param = True
        i += 1

        
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    
    study_name = 'optimize-CNN' # 最適化計算の名前
    study_name_path = study_name + '.db'
    y_dim = 3 # 変位の次元数
    memmap_dir = 'memmap' # デフォルトでmemmapを保存するディレクトリ
    first_divide = 0.2 # 8:2，学習データ:テストデータ
    second_devide = 0.25 # 6:2:2，訓練データ:検証データ:テストデータ
    best_model_name = 'best_model.h5'
    exist_best_model = os.path.exists(best_model_name)# 解析ディレクトリに学習モデルがあるかの確認．
    input_shape = (32, 32) # 入力データの形状，32[pixel]×32[pixel]
    
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 

    # 予測時にもドロップアウト層を有効にした学習モデルで再学習を行う．
    if interactive_mode:
        if input_str('モンテカルロドロップアウトの学習モデルにしますか．（予測時にもドロップアウト層を使うような学習モデル．y/n）>> ').lower().startswith('y'):
            monte_carlo_dropout = True
            if exist_best_model:
                interactive_mode = False
                if input_str('Optunaによる最適化モデルの探索を行いますか？(y/n)>> ').lower().startswith('y'):
                    pre_training_MC = True
                    interactive_mode = True
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
        memmap_dir = data_directory
    
    # 学習データの読み込み条件の指定．
    if interactive_mode or not batch_determination:
        if input_str('学習データをバッチごとに読み込みますか．（データが多いときはy，y/n）>> ').lower().strip().startswith('y'):
            load_split_batch = True
    
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
        n_trials = input_str('Optunaによる最適化の試行回数を指定してください．（指定しなければずっと計算を行う．）>> ')
        try:
            n_trials=int(n_trials) # str型をint型に変換
        except:
            n_trials=None
    
    # 前回までの最適化データの削除
    if interactive_mode:
        if input_str('前回までの最適化データがある場合，それを削除しますか．（y/n）>> ').lower().startswith('y'):
            reset_optimize = True
    if reset_optimize and not check_param:
        if os.path.exists(study_name_path):
            os.remove(study_name_path)

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
                dataset = MyDataset(y_dim, global_dict=globals().items())
                input_data = dataset.im2array(data_directory, n_jobs=n_jobs) # 入力データの画像 -> 配列に変換
                
                 # 出力データの読み込み
                output_path = glob(os.path.join(data_directory,'**', 'dataset_output.txt'), recursive=True)[0]
                output_data = get_input_output_from_file(output_path, top_skip = 1, input_columns = None, output_columns = range(y_dim), delimiter = None, encoding = 'UTF-8')
                
                print(u'総データ数： ' + str(output_data.shape[0]))
                
                # 学習用データとテストデータに分割
                x_learn, x_test_data, y_learn, y_test_data = train_test_split(input_data, output_data, test_size=first_divide)
                # 学習用データを訓練データと検証データに分割
                x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(x_learn, y_learn, test_size=second_devide)
                
                dataset.save_memmap((x_train_data, x_test_data, x_val_data, y_val_data, y_train_data, y_test_data), memmap_dir) # memmapファイルを保存
                data_size = [y_train_data[0], y_val_data[0], y_test_data[0]]
                # メモリの開放
                del input_data, output_data, x_test_data, y_test_data
                gc.collect()

            elif load_split_batch:
                print('訓練データをバッチごとに読み込みます．')
                # 訓練データをバッチごとに読み込むときのデータセット
                dataset = MyGeneratorDataset(y_dim=y_dim, n_jobs=n_jobs, global_dict=globals().items())
                data_set = dataset.get_paths(data_directory)
                print(u'総データ数： ' + str(len(data_set)))
                learning_data, test_data = train_test_split(data_set, test_size=first_divide) # データセットを学習用データとテストデータに分割
                train_data, val_data = train_test_split(learning_data, test_size=second_devide) # 学習用データを訓練データと検証データに分割
                data_size = dataset.generate_memmap((train_data, val_data, test_data), memmap_dir)
        
        elif use_memmap:
            data_size = []
            for i in ['train_data', 'val_data', 'test_data']:
                y_path = 'y_' + i + '.npy'
                y_memmap = np.memmap(filename=os.path.join(memmap_dir, y_path), dtype=np.float32, mode='r')
                data_size.append(y_memmap.reshape(-1, y_dim).shape[0]) # memmapファイルを利用して学習を行う．
                del y_memmap
        am_list = calc_am(memmap_dir, y_dim, output_num, output_axis)
        if monte_carlo_dropout and not pre_training_MC:
            restudy_by_monte_carlo_dropout(best_model_name, am_list, load_split_batch, memmap_dir, y_dim, output_num, output_axis, data_size, dropout_rate, study)
        else:
            study.optimize(objective, timeout=time_out, n_trials=n_trials)
        if monte_carlo_dropout and pre_training_MC:
            restudy_by_monte_carlo_dropout(best_model_name, am_list, load_split_batch, memmap_dir, y_dim, output_num, output_axis, data_size, dropout_rate, study)
    
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