#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:44:31
# read_data.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# テキストファイルから入力データ，出力データを取り出す関数，掲示板参照
def get_input_output_from_file(file_name, top_skip = 0, input_columns = (0,), output_columns = (1,),
    delimiter = None, encoding = 'UTF-8'):
    # encoding = 'UTF-8' | 'CP932'
    if input_columns is None:
        input_columns = tuple() # empty tuple
    elif not hasattr(input_columns, '__iter__'):
        input_columns = (input_columns,)
    if output_columns is None:
        output_columns = tuple() # empty tuple
    elif not hasattr(output_columns, '__iter__'):
        output_columns = (output_columns,)
    input = []
    output = []
    with open(file_name, 'r', encoding = encoding) as f:
        for i in range(top_skip):
            next(f)
        for line in f:
            line = line.split(delimiter)
            input.append([float(line[i]) for i in input_columns])
            output.append([float(line[i]) for i in output_columns])
    input = np.array(input, dtype = 'float32')
    output = np.array(output, dtype = 'float32')
    if input.shape[-1] == 1:
        input = input.reshape(input.shape[:-1])
    if output.shape[-1] == 1:
        output = output.reshape(output.shape[:-1])
    return output

def read_ymemmap(filename, dtype=np.float32, mode='r', y_dim=3, output_num=2, output_axis=0):
    y_memmap = np.memmap(filename=filename, dtype=dtype, mode=mode).reshape(-1, y_dim)
    return y_memmap[:, : output_num] if output_num != 1 else y_memmap[:, output_axis].reshape(-1,1)