#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-10-19 18:26:12
# convenient.py

import sys

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