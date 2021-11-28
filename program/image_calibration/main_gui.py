#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-25 18:25:58
# main_gui.py

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import sys
import os
from glob import glob
import subprocess

from read_data import get_calibration_info
from mk_projection_func import mk_projection_func

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

def get_calibration_file(event):
    try:
        calibration_info = get_calibration_info()
    except:
        calibration_info = glob(os.path.join('**', '*.cal'), recursive=True)
    if not isinstance(calibration_info, list): # list型じゃなかったら
        calibration_info = [calibration_info]
    calibration_files.set(calibration_info)

# リストボックスにおいて選択されているものを取得
def selection_listbox(listbox):
    assert listbox.size() != 0 # リストボックスが空のとき
    if listbox.size() == 1: # リストボックス内が1個のとき
        index = 0
    else:
        index = listbox.curselection()[0]
    return listbox.get(index)

# テキストエディターで開く
def open_calibration_file():
    calibration_file = selection_listbox(calibration_files_listbox)
    if sys.platform == 'win32':
            os.startfile(calibration_file) # ファイルに関連付けされたアプリで開く
    elif sys.platform == 'darwin':
        subprocess.call(u'open "%s"' % calibration_file, shell = True)
    else:
        subprocess.call(u'xdg-open "%s"' % calibration_file, shell = True)

def mk_projection_func_view(projection_func_coef):
    # 係数の数から次数を逆算
    order = order_sum = 0
    while order_sum != (len(projection_func_coef)//2):
        order += 1
        order_sum += order
    
    coef_num = 0
    for i in range(order + 1):
        coef_num += i

    img2phys_x = []
    img2phys_y = []
    phys2img_x = []
    phys2img_y = []
    x_index = order
    x_index_list = []
    while x_index != 0:
        for i in range(x_index):
            x_index_list.append(order - x_index)
        x_index -= 1
    y_index = order
    y_index_list = []
    while y_index != 0:
        for i in range(y_index):
            y_index_list.append(i)
        y_index -= 1
    for index, value in enumerate(projection_func_coef):
        value_x = value[0]
        value_y = value[1]

        try:
            x_index = x_index_list[index]
            y_index = y_index_list[index]
        except:
            x_index = x_index_list[index - coef_num]
            y_index = y_index_list[index - coef_num]
        assert (x_index + y_index) < order # x, y の指数の和が次数を超えたらエラー
        X = ''
        for i in range(x_index):
            X += '*X'
        Y = ''
        for i in range(y_index):
            Y += '*Y'
        if index < coef_num:
            img2phys_x.append('({}){}{}'.format(value[0], X, Y))
            img2phys_y.append('({}){}{}'.format(value[1], X, Y))
        else:
            phys2img_x.append('({}){}{}'.format(value[0], X, Y))
            phys2img_y.append('({}){}{}'.format(value[1], X, Y))
    return ' + '.join(img2phys_x), ' + '.join(img2phys_y), ' + '.join(phys2img_x), ' + '.join(phys2img_y)

def mk_projection_func_and_view():
    calibration_file = selection_listbox(calibration_files_listbox)
    print(calibration_file)
    projection_func, projection_func_coef = mk_projection_func(calibration_file)
    view_list = mk_projection_func_view(projection_func_coef)

    # テキストボックスを編集可能に変更
    img2phys_x_textbox.configure(state='normal')
    img2phys_y_textbox.configure(state='normal')
    phys2img_x_textbox.configure(state='normal')
    phys2img_y_textbox.configure(state='normal')

    # テキストボックスをクリア
    try:
        img2phys_x_textbox.delete(0, tk.END)
        img2phys_y_textbox.delete(0, tk.END)
        phys2img_x_textbox.delete(0, tk.END)
        phys2img_y_textbox.delete(0, tk.END)
    except:
        pass

    # テキストボックスに文字を入力
    img2phys_x_textbox.insert(tk.END, view_list[0])
    img2phys_y_textbox.insert(tk.END, view_list[1])
    phys2img_x_textbox.insert(tk.END, view_list[2])
    phys2img_y_textbox.insert(tk.END, view_list[3])
    
    # テキストボックスを編集不可に変更
    img2phys_x_textbox.configure(state='disabled')
    img2phys_y_textbox.configure(state='disabled')
    phys2img_x_textbox.configure(state='disabled')
    phys2img_y_textbox.configure(state='disabled')




if __name__ == '__main__':
    cwd = os.getcwd()
    button_frm_list = []


    # メインウィンドウ
    main_win = tk.Tk()
    main_win.title('動画を校正する（現在の解析ディレクトリ：{}）'.format(cwd))
    main_win.geometry('1000x400')

    # ペインドウィンドウの生成
    pw = tk.PanedWindow(main_win, sashrelief = tk.RAISED, sashwidth = 4)
    pw.pack(expand = True, fill = tk.BOTH)
    
# パラメータ
    cwd = tk.StringVar() # 解析ディレクトリ
    calibration_files = tk.StringVar() # キャリブレーション結果のファイル，複数の場合はlist型
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
    get_projection_func_btn.bind('<1>', get_calibration_file, '+') # ボタンを左クリックしたときの動作，追加処理
    button_frm_list.append(get_projection_func_btn)
    
    # フレームの作成
    get_projection_func_frm = tk.LabelFrame(pw, text=text)

    # ウィジェット作成（キャリブレーション結果のファイルを表示）
    get_projection_func_label = tk.Label(get_projection_func_frm, text='読み込むキャリブレーションファイル')
    calibration_files_listbox = tk.Listbox(get_projection_func_frm, listvariable=calibration_files)
    get_projection_func_btn = tk.Button(get_projection_func_frm, text='開く', command=open_calibration_file)

    # ウィジェット作成（実行ボタン）
    app_projection_func_btn = ttk.Button(get_projection_func_frm, text="投影関数の生成", command=mk_projection_func_and_view)

    # ウィジェット作成（投影関数を表示）
    img2phys_textbox_label = tk.Label(get_projection_func_frm, text='投影関数 \n画面座標（X,Y）→ 物理座標（X,Y）')
    phys2img_textbox_label = tk.Label(get_projection_func_frm, text='投影関数 \n物理座標（X,Y）→ 画面座標（X,Y）')
    img2phys_x_textbox = tk.Text(get_projection_func_frm, height=4, padx=5, pady=5)
    img2phys_y_textbox = tk.Text(get_projection_func_frm, height=4, padx=5, pady=5)
    phys2img_x_textbox = tk.Text(get_projection_func_frm, height=4, padx=5, pady=5)
    phys2img_y_textbox = tk.Text(get_projection_func_frm, height=4, padx=5, pady=5)

    # テキストボックスを編集不可に変更
    img2phys_x_textbox.configure(state='disabled')
    img2phys_y_textbox.configure(state='disabled')
    phys2img_x_textbox.configure(state='disabled')
    phys2img_y_textbox.configure(state='disabled')

    # ウィジェットの配置
    get_projection_func_label.grid(column=0, row=0, pady=10)
    calibration_files_listbox.grid(column=1, row=0, sticky=tk.EW, padx=5)
    get_projection_func_btn.grid(column=2, row=0, padx=5, pady=5)
    app_projection_func_btn.grid(column=1, row=2) # 投影関数生成ボタン
    img2phys_textbox_label.grid(column=0, row=4)
    phys2img_textbox_label.grid(column=0, row=8)
    img2phys_x_textbox.grid(column=1, row=4, sticky=tk.EW)
    img2phys_y_textbox.grid(column=1, row=6, sticky=tk.EW)
    phys2img_x_textbox.grid(column=1, row=8, sticky=tk.EW)
    phys2img_y_textbox.grid(column=1, row=10, sticky=tk.EW)

    # 列，行の最小幅を指定
    col_count, row_count = get_projection_func_frm.grid_size()
    # 列の最小幅
    for col in range(col_count):
        get_projection_func_frm.grid_columnconfigure(col, minsize=10)
    # 行の最小幅
    for row in range(row_count):
        get_projection_func_frm.grid_rowconfigure(row, minsize=10)

# ウィジェット，フレーム（お試し校正）の作成
    text = 'お試し校正'
    # button_frm 用のボタン
    trial_calibration_btn = tk.Button(button_frm, text=text, relief=tk.FLAT) # ボタンの作成
    trial_calibration_btn.bind('<1>', raise_frm) # ボタンを左クリックしたときの動作
    button_frm_list.append(trial_calibration_btn)

    # フレームの作成
    trial_calibration_frm = tk.LabelFrame(pw, text=text)
    
    

# ウィジェット，フレーム（校正する動画の選択）の作成
    text = '校正する動画の選択'
    # button_frm 用のボタン
    set_video_btn = tk.Button(button_frm, text=text, relief=tk.FLAT) # ボタンの作成
    set_video_btn.bind('<1>', raise_frm) # ボタンを左クリックしたときの動作
    button_frm_list.append(set_video_btn)

    # フレームの作成
    set_video_frm = tk.LabelFrame(pw, text=text)

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