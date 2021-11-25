#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# main_gui.py

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

def ask_folder():
    ''' 参照ボタンの動作
    '''
    path = filedialog.askdirectory()
    cwd.set(path)

def chdir():
    os.chdir(cwd.get())
    main_win.title('動画を校正する（現在の解析ディレクトリ：{}）'.format(cwd.get()))

def raise_frm(event):
    text = event.widget["text"]
    print(text)
    for key, value in globals().items():
        try:
            if text == value['text'] and 'frm' in key:
                if len(pw.panes()) != 1:
                    pw.forget(pw.panes()[-1])
                value.grid(column=0, row=0, sticky=tk.NSEW, padx=5, pady=10)
                pw.add(value)
                value.columnconfigure(1, weight=1)
        except:
            pass



if __name__ == '__main__':
    cwd = os.getcwd()
    button_frm_list = []


    # メインウィンドウ
    main_win = tk.Tk()
    main_win.title('動画を校正する（現在の解析ディレクトリ：{}）'.format(cwd))
    main_win.geometry('1000x200')

    # ペインドウィンドウの生成
    pw = tk.PanedWindow(main_win, sashrelief = tk.RAISED, sashwidth = 4)
    pw.pack(expand = True, fill = tk.BOTH)
    
# パラメータ
    cwd = tk.StringVar()
    # folder_path = tk.StringVar()

 # ボタンフレーム
    button_frm = tk.LabelFrame(pw, text='menu')
    button_frm.grid(column=0, row=0, sticky=tk.NSEW)
    pw.add(button_frm)

# ウィジェット，フレーム（解析ディレクトリの変更）の作成
    text = '解析ディレクトリの変更'
    # button_frm 用のボタン
    chdir_btn = tk.Button(button_frm, text=text, relief=tk.FLAT) # ボタンの作成
    chdir_btn.bind('<1>', raise_frm) # ボタンを左クリックしたときの動作
    button_frm_list.append(chdir_btn)
    
    # フレームの作成
    chdir_frm = tk.LabelFrame(pw, text=text)

    # ウィジェット作成（解析ディレクトリ指定用）
    chdir_label = tk.Label(chdir_frm, text='解析ディレクトリ指定')
    chdir_box = tk.Entry(chdir_frm, textvariable=cwd)
    chdir_btn = tk.Button(chdir_frm, text='参照', command=ask_folder)

    # ウィジェット作成（実行ボタン）
    app_chdir_btn = ttk.Button(chdir_frm, text="ディレクトリの移動", command=chdir)

    # ウィジェットの配置
    chdir_label.grid(column=0, row=0, pady=10)
    chdir_box.grid(column=1, row=0, sticky=tk.EW, padx=5)
    chdir_btn.grid(column=2, row=0)
    app_chdir_btn.grid(column=1, row=1)

# ウィジェット，フレーム（投影関数の取得）の作成
    text = '投影関数の取得'
    # button_frm 用のボタン
    get_projection_func_btn = tk.Button(button_frm, text=text, relief=tk.FLAT) # ボタンの作成
    get_projection_func_btn.bind('<1>', raise_frm) # ボタンを左クリックしたときの動作
    button_frm_list.append(get_projection_func_btn)
    
    # フレームの作成
    get_projection_func_frm = tk.LabelFrame(pw, text=text)

# ウィジェット，フレーム（校正する動画の選択）の作成
    text = '校正する動画の選択'
    # button_frm 用のボタン
    set_movie_btn = tk.Button(button_frm, text=text, relief=tk.FLAT) # ボタンの作成
    set_movie_btn.bind('<1>', raise_frm) # ボタンを左クリックしたときの動作
    button_frm_list.append(set_movie_btn)

    # フレームの作成
    set_movie_frm = tk.LabelFrame(pw, text=text)

    # # ウィジェット（フォルダ名）
    # folder_label = tk.Label(main_frm, text='フォルダ指定')
    # folder_box = tk.Entry(main_frm, textvariable=folder_path)
    # folder_btn = tk.Button(main_frm, text='参照', command=ask_folder)

    # # ウィジェット（並び順）
    # order_label = tk.Label(main_frm, text='並び順')
    # order_comb = ttk.Combobox(main_frm, values=['昇順', '降順'], width=10)
    # order_comb.current(0)

    # # ウィジェット（実行ボタン）
    # app_btn = tk.Button(main_frm, text='実行', command=app)

    # ウィジェットの配置(button_frm)
    for index, value in enumerate(button_frm_list):
        value.grid(column=0, row=index)
    # folder_btn.grid(column=2, row=0)
    # order_label.grid(column=0, row=1)
    # order_comb.grid(column=1, row=1, sticky=tk.W, padx=5)
    # app_btn.grid(column=1, row=2)

    # 配置設定
    main_win.columnconfigure(0, weight=1)
    main_win.rowconfigure(0, weight=1)

    main_win.mainloop()