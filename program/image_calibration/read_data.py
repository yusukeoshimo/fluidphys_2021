#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# read_data.py

import os
from glob import glob

# 校正の情報が入ったファイルのリストを取得
def get_calibration_info():
    calibration_info_list = glob(os.path.join('**', 'Calibration.cal'), recursive=True)
    assert calibration_info_list == 1 # ファイルが2以上あったらエラー
    return calibration_info_list[0]