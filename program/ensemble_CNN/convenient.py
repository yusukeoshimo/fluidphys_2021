#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-25 20:27:37
# convenient.py

import sys
import os
import shutil

def input_str(message):
    return (raw_input if sys.version_info.major <= 2 else input)(message)

def input_int(message):
    while True:
        try:
            x = int((raw_input if sys.version_info.major <= 2 else input)(message))
            return x
        except:
            pass

def input_float(message):
    while True:
        try:
            x = float((raw_input if sys.version_info.major <= 2 else input)(message))
            return x
        except:
            pass

def remove_dir(dir_name):
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)