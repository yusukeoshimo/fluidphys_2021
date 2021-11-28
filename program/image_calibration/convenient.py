#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-28 18:30:36
# convenient.py

import sys
import os
import shutil
import subprocess

def input_str(message):
    return (raw_input if sys.version_info.major <= 2 else input)(message).strip().strip("'")

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

def remake_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

# 書き込み専用の関数，mode は 'w'か'a'
def write_txt(file_name, mode, contents):
    with open(file_name, mode=mode) as f:
        f.write(contents)

# ファイルに関連付けされたアプリで開く
def open_file(path):
    if sys.platform == 'win32':
            os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.call(u'open "%s"' % path, shell = True)
    else:
        subprocess.call(u'xdg-open "%s"' % path, shell = True)