#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-04 21:14:39
# main.py

import sys
import os
import numpy as np
import time

from convenient import input_str, input_int, input_float, remake_dir

def check_output_axis(output_num):
    if output_num == 1:
        output_axis = input_int('出力の軸（推論したい速度方向）を入力してください．（x or y or z） >> ')
        # 出力データのaxis = -2におけるインデックスを決定．x, y, z の順番に並んでいると考えている．
        if output_axis.lower().strip().startswith('x'):
            output_axis = 0
        elif output_axis.lower().strip().startswith('y'):
            output_axis = 1
        elif output_axis.lower().strip().startswith('z'):
            output_axis = 2
        assert isinstance(output_axis, int) # output_axis がint型になってなければエラー
        return output_axis
    return None

# 書き込み専用の関数，mode は 'w'か'a'
def write_txt(file_name, mode, contents):
    with open(file_name, mode=mode) as f:
        f.write(contents)


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
                  u'-h[elp]                     -> ヘルプの表示．\n'
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
    if interactive_mode and not (model_learning or model_predict):
        if input_str('アンサンブル学習を行いますか．(y/n) >> ').lower().startswith('y'):
            model_learning = True
        if input_str('学習させたモデル群で推論を行いますか．(y/n) >> ').lower().startswith('y'):
            model_predict = True
    
    # モデル構造が指定されているのか確認，学習する際に必要
    if model_path is None and model_learning:
        model_path = input_str('ロードするモデルを指定してください．>> ')
        assert model_path != '' # 何も入力していない場合エラー
        assert os.path.exists(model_path) # ファイルが存在していない場合エラー

    # 推論に使うモデル群のディレクトリを指定
    if model_dir is None and model_predict and not model_learning: # 推論あり，学習なしの場合
        model_dir = input_str('推論に使うモデル群のディレクトリを指定してください．>> ')
        assert model_num != '' # 何も入力していない場合エラー
        assert os.path.exists(model_dir) # ファイルが存在していない場合エラー

    # 学習，推論に使用するデータ数が指定されているのか確認，学習／推論する際に必要
    if data_num is None:
        data_num = input_int('学習／推論に使うデータ数を指定してください．>> ')
        assert data_num != '' # 何も入力していない場合エラー

    # アンサンブル学習に使うモデル数を確認
    if model_num is None and model_learning:
        model_num = input_int('アンサンブル学習に使うモデル数を指定してください．>> ')
        assert model_num != '' # 何も入力していない場合エラー
    

    if model_path is not None:
        # アンサンブル学習を行うモデルを保存するディレクトリ
        savedir_ensemble_model = 'models_{}_datanum={}'.format(os.path.dirname(model_path).split(os.path.sep)[-1], data_num)
    #                                                                                          パスの区切り文字
    if model_dir is None and model_predict and model_learning: # 推論あり，学習ありの場合
        model_dir = savedir_ensemble_model

    if data_num > 0.5e07: # 学習データが50万以上の場合はバッチごとにデータを読み込む
        load_split_batch = True

    import tensorflow as tf
    from model_operation import model_load, reset_weights
    
    # モデルの出力数の確認
    if model_learning:
        model = model_load(model_path)
        output_num = model.output.shape[-1] # 出力数の抽出
        output_axis = check_output_axis(output_num)
    elif model_predict and not model_learning:
        # 拡張子が '.h5' のファイルのリストを作成
        models_path_list = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if '.h5' in f]
        model = model_load(models_path_list[0])
        output_num = model.output.shape[-1] # 出力数の抽出
        model_num = len(models_path_list) # 推論に使う学習モデルの数
        output_axis = check_output_axis(output_num)

    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    y_dim = 3 # 変位の次元数
    memmap_dir = 'memmap_' + str(data_num) # デフォルトでmemmapを保存するディレクトリ
    save_predict_result = 'predict_result.txt' # 推論結果を保存するファイル
    save_learning_result = 'learning_result' # 学習結果を保存するディレクトリ
    learning_time_save = os.path.join(save_learning_result, 'learning_time.txt') # 学習時間を保存するファイルのパス
    split_everything_to_train_validation = 0.2 # 8:2，訓練データ:検証データ
    split_rate = [split_everything_to_train_validation]
    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
    
    from model_train import model_train, calc_am
    from my_callbacks import LossHistory, CopyWeights
    from particle_image_with_fluid_func import mkdata

    if model_learning:
        remake_dir(savedir_ensemble_model)
        remake_dir(save_learning_result)

        # 学習時間を保存するファイルの作成
        write_txt(learning_time_save, 'w', '# モデル番号 学習時間     \n')
        learning_time_list = [] # 学習時間を保存する空のリスト
        
        for i in range(model_num):
            print('\n{}個目のモデルの学習データの作成\n'.format(i))
            ensemble_model = reset_weights(model) # 重み，バイアスの初期化
            data_size = mkdata(data_num, memmap_dir, y_dim, split_rate) # 学習データの作成，memmapファイルに変換
            am_list = calc_am(memmap_dir, y_dim, output_num, output_axis)

            #epoch毎にグラフ描画
            cb_figure = LossHistory(save_name='model_{}_learning_curve'.format(i), 
                                    save_dir=save_learning_result, 
                                    model_num=i, 
                                    output_num=output_num, 
                                    am_list=am_list
                                    )
            
            #使用するコールバックスの指定
            callbacks = [cb_figure]
            
            # 学習条件
            verbose = 1
            epochs = 100
            batch_size = 512 # Optunaの最適化結果を確認して最適なものに変更する必要あり

            print('\n{}個目のモデルの学習を開始\n'.format(i))
            time_start = time.time() # 学習開始時の時間を記録
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
                                                  )
            time_end = time.time() # 学習終了時の時間を記録
            learning_time = time_end - time_start # 学習時間
            learning_time_list.append(learning_time)
            # 学習時間をテキストファイルに書き出す
            write_txt(learning_time_save, 'a', '{} {}\n'.format(i, learning_time))
            # モデルの保存
            ensemble_model.save(os.path.join(savedir_ensemble_model, 'ensemble_model_{}.h5'.format(i)))
        # 学習時間の平均を計算
        learning_time_mean = np.mean(np.array(learning_time_list))
        # 学習時間の平均をテキストファイルに書き出す
        write_txt(learning_time_save, 'a', 'average time : {}\n'.format(learning_time_mean))
    
    if model_predict:
        # 推論用モデルのロード
        models_list = [] # 推論用モデルを保存するリスト
        for i in models_path_list:
            models_list.append(model_load(i))
        assert len(models_list) != 0 # 推論用モデルがなければエラー

        # 推論用データの作成
        data_size = mkdata(data_num, memmap_dir, y_dim) # 推論用データの作成，memmapファイルに変換
        if hasattr(data_size, '__iter__'):
            assert len(data_size) == 1 # data_size 内の要素が1つでなければエラー
            data_size = data_size[0]
        am_list = calc_am(memmap_dir, y_dim, output_num, output_axis)

        # 推論結果を保存するテキストデータを作成
        write_txt(os.path.join(save_predict_result), 'w',
                  '# モデル数：{}\n'.format(model_num) +
                  '# モデルのパス：{}\n'.format(model_dir if model_dir.startswith('C:') else '..\{}'.format(model_dir)) +
                  '# 推論したデータのパス：{}\n'.format(memmap_dir if memmap_dir.startswith('C:') else '..\{}'.format(memmap_dir)) +
                  '# 番号     平均     分散     \n')
        
        for i in range(data_size):
            # 入力データの取得
            X_MEMMAP_NAME = 'x_test_data'
            X = np.memmap(filename=os.path.join(memmap_dir, '{}.npy'.format(X_MEMMAP_NAME)), 
                          dtype=np.float32, 
                          mode='r',
                          shape=(data_size, 32, 32, 2)
                          )
            input_data = [np.expand_dims(X[i,:,:,0], 0), np.expand_dims(X[i,:,:,1], 0)]
            #  shape =            (1,32,32),                      (1,32,32)

            output_list = [] # 出力を入れる空のリストを作成
            for predict_model in models_list:
                predict_model.summary()
                history = predict_model.predict(x=input_data)
                output_list.append(history) # 出力をリストに追加
            output_array = np.array(output_list) # リストを配列に変換
            # output_array.shape = (推論したモデルの数，出力の数)
            assert isinstance(output_array, np.ndarray) # 配列に変換できない場合はエラー
            # 平均の計算
            # 移動距離の計算，move_distance = √(u_x^2 + u_y^2 + u_z^2)
            move_distance = np.sqrt(np.sum(np.square(output_array),-1).reshape(-1,1))
            output_mean = np.mean(move_distance)
            # 分散の計算，variance = 1/n * Σ(y-y_mean)^2
            output_variance = np.var(move_distance)

            # 推論結果を保存するテキストデータを作成
            write_txt(os.path.join(save_predict_result), 'a', '{} {} {}\n'.format(i, output_mean, output_variance))
