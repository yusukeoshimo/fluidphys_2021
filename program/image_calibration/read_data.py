#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-26 11:29:44
# read_data.py

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
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

def load_video(video_path):
    video_file = cv2.VideoCapture(video_path)
    assert video_file.isOpened() # 動画ファイルを読み込めていなければエラー
    return video_file

def get_video_path(video):
    if os.path.isfile(video): # ファイルかどうか確認
        video_path = [video] # リスト型
    else:
        video_path = []
        # ファイルだけを再帰的に取得
        file_paths = [f for f in glob(os.path.join('**', '*'), recursive=True) if os.path.isfile(f)]
        for file_path in file_paths:
            try:
                load_video(file_path).release() # 動画を閉じる．
                video_path.append(file_path)
            except:
                pass
    return video_path

# 画像をnumpy配列で読み出す関数，掲示板参照
def read_image(file_name, flags = cv2.IMREAD_GRAYSCALE, resize = None, show = False):
    # flags = cv2.IMREAD_COLOR | cv2.IMREAD_GRAYSCALE | ...
    # (see https://docs.opencv.org/4.2.0/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80)
    im = cv2.imread(file_name, flags)
    if flags == cv2.IMREAD_COLOR:
        im = im[:, :, (2, 1, 0)] # BGR -> RGB
    if resize is not None:
        if type(resize[0]) is int and type(resize[1]) is int:
            im = cv2.resize(im, dsize = resize)
        else:
            im = cv2.resize(im, dsize = None, fx = resize[0], fy = resize[1])
    if show:
        plt.clf()
        plt.imshow(im)
        plt.show()
    return im