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