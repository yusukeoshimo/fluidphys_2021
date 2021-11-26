#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-27 00:34:34
# calibration.py

import os
import numpy as np
import cv2
from scipy.interpolate import LinearNDInterpolator

from read_data import load_video

def calibration_image(frame, projection_func):
    # フラグ
    grayscale = False
    # 配列の形，縦幅，横幅の確認
    frame_shape = frame.shape
    height = frame_shape[0]
    width = frame_shape[1]

    if len(frame_shape) == 2:
        grayscale = True
    
    x = np.arange(width)
    y = np.arange(height).reshape(-1,1)

    # 画像座標から返還した物理座標を取得
    phys_x, phys_y, phys2img_x, phys2img_y = projection_func(x,y)
    # いらない変数を削除
    del phys2img_x, phys2img_y

    # 補間用に配列を返還
    phys_x = phys_x.reshape(-1,1)
    phys_y = phys_y.reshape(-1,1)
    if grayscale: # グレースケールの場合
        frame = frame.reshape(-1,1)
    else:
        frame = np.hstack((frame[:,:,0].reshape(-1,1), frame[:,:,1].reshape(-1,1), frame[:,:,2].reshape(-1,1)))

    # 変換した座標の範囲を元画像に合わせる
    phys_x_max = np.max(phys_x)
    phys_x_min = np.min(phys_x)
    phys_x = (phys_x - phys_x_min)/(phys_x_max-phys_x_min) * width

    phys_y_max = np.max(phys_y)
    phys_y_min = np.min(phys_y)
    phys_y = (phys_y - phys_y_min)/(phys_y_max-phys_y_min) * height


    # 輝度の補間関数を生成
    # LinearNDInterpolator(points, values)
    #                    points   |     values
    #                  [[x0, y0]  | [[luminance_0],
    #                   [x1, y1]  |  [luminance_1],
    #                   [x2, y2]  |  [luminance_2],
    #                   ...    ,  |  ...          ,
    #                   [xm, ym]] |  [luminance_m]]
    #                                              
    luminance_interpolation_func = LinearNDInterpolator(np.hstack((phys_x, phys_y)), frame)
    
    # 補間用の座標を作成
    x_array = (np.arange(width)*(np.ones(height).reshape(-1,1))).reshape(-1,1)
    y_array = ((np.arange(height).reshape(-1,1))*np.ones(width)).reshape(-1,1)
    return luminance_interpolation_func(np.hstack((x_array, y_array))).reshape((height, width) if grayscale else (height, width, 3))

# 読み込んだ動画をもとに新しい動画ファイルを作成
def mk_writer(path, video_savedir):
    video = load_video(path)
    # 幅と高さを取得
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    #総フレーム数を取得
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    #フレームレート(1フレームの時間単位はミリ秒)の取得
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    video.release() # ファイルを閉じる

    # 保存用
    fmt = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(os.path.join(video_savedir, 'calibrated_{}'.format(os.path.basename(path))), fmt, frame_rate, size)

def calibration_video(video_path, video_savedir, projection_func):
    for path in video_path:
        video = load_video(path) # 動画ファイルを読み込む
        writer = mk_writer(path, video_savedir) # 書き込み用の動画ファイルを作成
        while True:
            try:
                ret, frame = video.read() # 1フレーム分のndarrayがframeに入る
                if not ret: # 最後のフレームまでいったら
                    break

                calibrated_frame = calibration_image(frame, projection_func)
                writer.write(calibrated_frame) # 画像を1フレーム分として書き込み
            except:
                pass
        video.release() # ファイルを閉じる
        writer.release() # ファイルを閉じる

if __name__ == '__main__':
    from convenient import input_str
    from read_data import read_image, get_calibration_info
    from mk_projection_func import mk_projection_func
    import time
    import matplotlib.pyplot as plt

    # 校正の情報が入ったファイルのパスを取得
    calibration_info = get_calibration_info()
    # 投影関数（関数オブジェクト），投影関数の係数（リスト型）を取得
    projection_func, projection_func_coef = mk_projection_func(calibration_info)

    image_path = input_str('校正を行う画像を指定してください．>> ')
    image = read_image(image_path, flags = cv2.IMREAD_COLOR, show = True)
    # flags = cv2.IMREAD_COLOR | cv2.IMREAD_GRAYSCALE

    start_time = time.time()
    calibrated_frame = calibration_image(image, projection_func)
    end_time = time.time()
    print('finish!! time = {}'.format(end_time - start_time))
    # 画像の保存
    save_image = 'calibrated_{}'.format(os.path.basename(image_path))
    cv2.imwrite(save_image, calibrated_frame)
    # 保存した画像の確認
    read_image(save_image, flags = cv2.IMREAD_COLOR, show = True)