#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# main.py

import sys
import os
import numpy as np
from glob import glob
import time

from read_data import get_calibration_info, get_video_path
from convenient import input_str, remake_dir
from calibration import calibration_video

if __name__ == '__main__':
    cwd = os.getcwd() # カレントディレクトリの取得
    video_savedir = 'calibrated_video' # 校正した動画を保存するディレクトリ

    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        if sys.argv[i].lower().startswith('-h'): # ヘルプの表示
            print(u'使い方: python {} -g[rid] grid_image .... \n'.format(os.path.basename(sys.argv[0])) +
                  u'---- オプション ----\n' +
                  u'-v[ideo] video             -> 校正する動画を video で指定，動画の入ったディレクトリでも可．\n' +
                  u'-h[elp]                     -> ヘルプの表示．\n'
                  )
            sys.exit(0) # 正常終了, https://www.sejuku.net/blog/24331
        elif sys.argv[i].lower().startswith('-v'): # 校正する動画の指定
            i += 1
            video = sys.argv[i]
        i +=1
    
    if interactive_mode:
        video = input_str('校正する動画を指定してください．複数の場合はディレクトリを指定してください．>> ')
        assert os.path.exists(video_path) # 指定したパスがなければエラー

    # 校正の情報が入ったファイルのパスを取得
    calibration_info = get_calibration_info()
    # 投影関数（関数オブジェクト），投影関数の係数（リスト型）を取得
    projection_func, projection_func_coef = mk_projection_func(calibration_info)

    # 動画のパスを取得（型はリスト）
    video_path = get_video_path(video)

    # 動画を保存するディレクトリの作成
    remake_dir(video_savedir)

    calibration_video(video_path, video_savedir, projection_func)