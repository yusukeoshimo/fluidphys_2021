#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# main.py

import sys
import os
import numpy as np
from glob import glob
import time

from read_data import get_calibration_info
from convenient import input_str

if __name__ == '__main__':
    cwd = os.getcwd() # カレントディレクトリの取得

    i = 1
    while i < len(sys.argv):
        interactive_mode = False
        if sys.argv[i].lower().startswith('-h'): # ヘルプの表示
            print(u'使い方: python {} -g[rid] grid_image .... \n'.format(os.path.basename(sys.argv[0])) +
                  u'---- オプション ----\n' +
                  u'-m[ovie] movie             -> 校正する動画を movie で指定，動画の入ったディレクトリでも可．\n' +
                  u'-h[elp]                     -> ヘルプの表示．\n'
                  )
            sys.exit(0) # 正常終了, https://www.sejuku.net/blog/24331
        elif sys.argv[i].lower().startswith('-m'): # 校正する動画の指定
            i += 1
            movie = sys.argv[i]
        i +=1
    
    if interactive_mode:
        movie = input_str('校正する動画を指定してください．複数の場合はディレクトリを指定してください．>> ')
        assert os.path.exists(movie_path) # 指定したパスがなければエラー

    # 校正の情報が入ったファイルのパスを取得
    calibration_info = get_calibration_info()
    # 投影関数（関数オブジェクト），投影関数の係数（リスト型）を取得
    projection_func, projection_func_coef = mk_projection_func(calibration_info)

    # 動画のパスを取得（型はリスト）
    movie_path = get_movie_path(movie)
    