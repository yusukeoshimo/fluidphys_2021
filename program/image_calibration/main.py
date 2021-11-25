#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# main.py

import sys
import os
from glob import glob
import numpy as np
import time

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
    

    # 校正の情報が入ったファイルのリストを取得
    calibration_info_list = glob(os.path.join('**', 'Calibration.cal'), recursive=True)
    assert calibration_info_list == 1 # ファイルが2以上あったらエラー
    calibration_info = calibration_info_list[0]
