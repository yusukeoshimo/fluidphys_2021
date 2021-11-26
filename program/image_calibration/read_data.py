#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-26 11:29:44
# read_data.py

import os
from glob import glob

# 校正の情報が入ったファイルのパスを取得
def get_calibration_info():
    calibration_info_list = glob(os.path.join('**', 'Calibration.cal'), recursive=True)
    assert len(calibration_info_list) == 1, 'calibration_info_list = {}'.format(calibration_info_list) # ファイルが2以上あったらエラー
    return calibration_info_list[0]

def read_text_data(text, search_word, split_str):
    with open(calibration_info, 'r') as f: # 読み取りモード
        for line in f: # 1行ずつ読む
            if line.strip().lower().startswith(search_word):
                return line.strip().split(split_str)[-1]
