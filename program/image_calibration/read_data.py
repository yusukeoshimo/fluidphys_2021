#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-26 11:29:44
# read_data.py

import os
from glob import glob
import cv2

# 校正の情報が入ったファイルのパスを取得
def get_calibration_info():
    calibration_info_list = glob(os.path.join('**', '*.cal'), recursive=True)
    assert len(calibration_info_list) == 1, 'calibration_info_list = {}'.format(calibration_info_list) # ファイルが2以上あったらエラー
    return calibration_info_list[0]

def read_text_data(text, search_word, split_str):
    with open(text, 'r') as f: # 読み取りモード
        for line in f: # 1行ずつ読む
            if line.strip().lower().startswith(search_word):
                return line.strip().split(split_str)[-1]

def load_movie(movie_path):
    movie_file = cv2.VideoCapture(movie_path)
    assert movie_file.isOpened() # 動画ファイルを読み込めていなければエラー
    return movie_file

def get_movie_path(movie):
    if os.path.isfile(movie): # ファイルかどうか確認
        movie_path = [movie] # リスト型
    else:
        movie_path = []
        # ファイルだけを再帰的に取得
        file_paths = [f for f in glob(os.path.join('**', '*'), recursive=True) if os.path.isfile(f)]
        for file_path in file_paths:
            try:
                load_movie(file_path).release() # 動画を閉じる．
                movie_path.append(file_path)
            except:
                pass
    return movie_path