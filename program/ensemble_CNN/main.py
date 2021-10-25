#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-22 13:01:43
# main.py

import sys
import os
import multiprocessing
from multiprocessing import Pool

from convenient import input_str, input_int, input_float
from particle_image_with_fluid_func import mkdata
from my_callbacks import LossHistory

if __name__ == '__main__':
    # カレントディレクトリの取得
    cwd = os.getcwd()

    # 後のif分判定に使うフラグ
    interactive_mode = True
    model_learning = False
    model_predict = False
    load_split_batch = False
    
    # プレースホルダー
    model_path = None
    data_num = None
    model_num = None
    model_dir = None
    output_axis = None

    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        if sys.argv[i].lower().startswith('-h'): # ヘルプの表示
            print(u'使い方: python {}'.format(os.path.basename(sys.argv[0])) +
                  u' -m[odel] model_path .... \n' +
                  u'---- オプション ----\n' +
                  u'-m[odel] model_path         -> アンサンブル学習に使うモデルを model_path で指定．\n' +
                  u'-d[ata_number] data_num     -> 学習またはテストに使うデータ数を data_num で指定．\n' +
                  u'-l[earning] model_num       -> アンサンブル学習の実行．学習に使うモデル数を model_num で指定．\n' +
                  u'-p[redict] [model_dir]      -> 推論を実行．推論に使うモデル群を model_dir で指定．\n' +
                  u'                               学習と推論を連続で行いたいときは，model_dir を指定しない．\n' +
                  u'-h[elp]                     -> ヘルプの表示．\n' +
                  )
            sys.exit(0) # 正常終了, https://www.sejuku.net/blog/24331
        if sys.argv[i].lower().startswith('-m'): # ロードするモデルの指定
            i += 1
            model_path = sys.argv[i]
        elif sys.argv[i].lower().startswith('-d'): # データ数の指定
            i += 1
            data_num = sys.argv[i]
        elif sys.argv[i].lower().startswith('-l'): # モデルの学習を行う
            model_learning = True
            i += 1
            model_num = sys.argv[i]
        elif sys.argv[i].lower().startswith('-p'): # モデルによる推論を行う
            model_predict = True
            if not sys.argv[i].startswith('-'):
                i += 1
                model_dir = sys.argv[i]
        i += 1

    # 実行方法（学習／推論）の指定．インタラクティブまたは，指定していないとき．
    if interactive_mode and (model_learning or model_predict):
        if input_str('アンサンブル学習を行いますか．(y/n) >> ').lower().startswith('y'):
            model_learning = True
        if input_str('学習させたモデル群で推論を行いますか．(y/n) >> ').lower().startswith('y'):
            model_predict = True
    
    # モデル構造が指定されているのか確認，学習する際に必要
    if model_path is None and model_learning:
        model_path = input_str('ロードするモデルを指定してください．>> ')
        assert model_path != '' # 何も入力していない場合エラー
        assert os.paht.exists(model_path) # ファイルが存在していない場合エラー

    # 学習，推論に使用するデータ数が指定されているのか確認，学習／推論する際に必要
    if data_num is None:
        data_num = input_int('学習／推論に使うデータ数を指定してください．>> ')
        assert data_num != '' # 何も入力していない場合エラー

    # アンサンブル学習に使うモデル数を確認
    if model_num is None and model_learning:
        model_num = input_int('アンサンブル学習に使うモデル数を指定してください．>> ')
        assert model_num != '' # 何も入力していない場合エラー
    
    # 推論に使うモデル群のディレクトリを指定，
    if model_dir is None and model_predict and not model_learning:
        model_dir = input_int('推論に使うモデル群のディレクトリを指定してください．>> ')
        assert model_num != '' # 何も入力していない場合エラー
        assert os.paht.exists(model_dir) # ファイルが存在していない場合エラー

    if data_num > 0.5e07: # 学習データが50万以上の場合はバッチごとにデータを読み込む
        load_split_batch = True

    from model_operation import model_load, reset_weights
    
    # モデルの出力数の確認
    if model_learning:
        model = model_load(model_path)
        model = reset_weights(model) # 重み，バイアスの初期化
        output_num = model.output.shape[-1] # 出力数の抽出
        if output_num == 1:
            output_axis = input_int('出力の軸（推定したい速度方向）を入力してください．（x or y or z） >> ')
            # 出力データのaxis = -2におけるインデックスを決定．x, y, z の順番に並んでいると考えている．
            if output_axis.lower().strip().startswith('x'):
                output_axis = 0
            elif output_axis.lower().strip().startswith('y'):
                output_axis = 1
            elif output_axis.lower().strip().startswith('z'):
                output_axis = 2
            assert isinstance(output_axis, int) # output_axis がint型になってなければエラー

    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    y_dim = 3 # 変位の次元数
    memmap_dir = 'memmap_' + str(data_num) # デフォルトでmemmapを保存するディレクトリ
    split_everything_to_train_validation = 0.2 # 8:2，訓練データ:検証データ
    # アンサンブル学習を行うモデルを保存するディレクトリ
    savedir_ensemble_model = os.path.dirname(model_path).split(os.path.sep)[-1] + '_datanum={}'.format(data_num)
    #                                                         パスの区切り文字
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 

    if model_learning:
        for i in range(model_num):
            ensemble_model = model # ロードしたモデル構造
            data_size = mkdata(data_num, memmap_dir) # 学習データの作成，memmapファイルに変換
            am_list = calc_am(memmap_dir, y_dim, output_num, output_axis)

            #epoch毎にグラフ描画
            cb_figure = LossHistory(save_name='model_{}_learning_curve'.format(i), 
                                    save_dir=savedir_ensemble_model, 
                                    model_num=i, 
                                    output_num=output_num, 
                                    am_list=am_list
                                    )
            
            #使用するコールバックスの指定
            callbacks = [cb_figure]
            
            # 学習条件
            verbose = 1
            epochs = 100
            batch_size = 512

            # モデルの学習
            ensemble_model, history = model_train(model=ensemble_model, 
                                                  verbose=verbose, 
                                                  epochs=epochs, 
                                                  batch_size=batch_size, 
                                                  callbacks=callbacks, 
                                                  load_split_batch=load_split_batch, 
                                                  memmap_dir=memmap_dir, 
                                                  y_dim=y_dim, 
                                                  output_num=output_num, 
                                                  output_axis=output_axis, 
                                                  data_size=data_size
                                                  ):
            
            # モデルの保存
            ensemble_model.save(os.path.join(savedir_ensemble_model, 'ensemble_model_{}.h5'.format(i)))
