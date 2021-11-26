#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-11-26 13:34:53
# processing_data.py

def read_text_data(text, search_word, split_str):
    with open(calibration_info, 'r') as f: # 読み取りモード
        for line in f: # 1行ずつ読む
            if line.strip().lower().startswith(search_word):
                return line.strip().split(split_str)[-1]

# 投影関数の係数を取得
def get_projection_func_coef(calibration_info):
    projection_func_coef = [] # 空のリスト
    with open(calibration_info, 'r') as f:
        line = f.readline()
        while not line.strip().lower().startswith('order'):
            line = f.readline()
        line = f.readline()
        while not line.strip().lower().startswith('end'):
            projection_func_coef.append([float(i) for i in line.strip().split(',')])
            line = f.readline()
    try:
        while True:
            projection_func_coef.remove([0.0, 0.0])
    except:
        return projection_func_coef


# 投影関数
def mk_projection_func(calibration_info):
    width = read_text_data(calibration_info, 'imwx', '=')
    height = read_text_data(calibration_info, 'imwy', '=')
    order = int(read_text_data(calibration_info, 'order', '='))
    projection_func_coef = get_projection_func_coef(calibration_info)
    coef_num = 0
    for i in range(order + 1):
        coef_num += i
    assert coef_num == len(projection_func_coef)/2, 'coef_num = {}, len(projection_func_coef)/2 = {}'.format(coef_num, len(projection_func_coef)/2) # 係数の数が一致してないとエラー
    def projection_func(x, y):
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

            if index < coef_num:
                img2phys_x.append(value[0]*x**x_index*y**y_index)
                img2phys_y.append(value[1]*x**x_index*y**y_index)
            else:
                phys2img_x.append(value[0]*x**x_index*y**y_index)
                phys2img_y.append(value[1]*x**x_index*y**y_index)
        return sum(img2phys_x), sum(img2phys_y), sum(phys2img_x), sum(phys2img_y)
    return projection_func #, projection_func_coef # 確認時使用

if __name__ == '__main__':
    from read_data import get_calibration_info
    calibration_info = get_calibration_info()
    projection_func = mk_projection_func(calibration_info)
    # projection_func, projection_func_coef = mk_projection_func(calibration_info) # 確認時使用

    from convenient import input_str
    x, y = [ int(i) for i in input_str('移動させる座標を指定してください．（ x, y ）>> ').strip().split(',')]

    new_coordinate = projection_func(x, y)
    for index, value in enumerate(['img2phys_x', 'img2phys_y', 'phys2img_x', 'phys2img_y']):
        print('{} = {}'.format(value, new_coordinate[index]))

    
    # 以下，確認用
    # img2phys_x = sum([projection_func_coef[0][0]*x**0*y**0,
    #                   projection_func_coef[1][0]*x**0*y**1,
    #                   projection_func_coef[2][0]*x**0*y**2,
    #                   projection_func_coef[3][0]*x**0*y**3,
    #                   projection_func_coef[4][0]*x**1*y**0,
    #                   projection_func_coef[5][0]*x**1*y**1,
    #                   projection_func_coef[6][0]*x**1*y**2,
    #                   projection_func_coef[7][0]*x**2*y**0,
    #                   projection_func_coef[8][0]*x**2*y**1,
    #                   projection_func_coef[9][0]*x**3*y**0
    #                   ])
    # img2phys_y = sum([projection_func_coef[0][1]*x**0*y**0,
    #                   projection_func_coef[1][1]*x**0*y**1,
    #                   projection_func_coef[2][1]*x**0*y**2,
    #                   projection_func_coef[3][1]*x**0*y**3,
    #                   projection_func_coef[4][1]*x**1*y**0,
    #                   projection_func_coef[5][1]*x**1*y**1,
    #                   projection_func_coef[6][1]*x**1*y**2,
    #                   projection_func_coef[7][1]*x**2*y**0,
    #                   projection_func_coef[8][1]*x**2*y**1,
    #                   projection_func_coef[9][1]*x**3*y**0
    #                   ])
    # phys2img_x = sum([projection_func_coef[10][0]*x**0*y**0,
    #                   projection_func_coef[11][0]*x**0*y**1,
    #                   projection_func_coef[12][0]*x**0*y**2,
    #                   projection_func_coef[13][0]*x**0*y**3,
    #                   projection_func_coef[14][0]*x**1*y**0,
    #                   projection_func_coef[15][0]*x**1*y**1,
    #                   projection_func_coef[16][0]*x**1*y**2,
    #                   projection_func_coef[17][0]*x**2*y**0,
    #                   projection_func_coef[18][0]*x**2*y**1,
    #                   projection_func_coef[19][0]*x**3*y**0
    #                   ])
    # phys2img_y = sum([projection_func_coef[10][1]*x**0*y**0,
    #                   projection_func_coef[11][1]*x**0*y**1,
    #                   projection_func_coef[12][1]*x**0*y**2,
    #                   projection_func_coef[13][1]*x**0*y**3,
    #                   projection_func_coef[14][1]*x**1*y**0,
    #                   projection_func_coef[15][1]*x**1*y**1,
    #                   projection_func_coef[16][1]*x**1*y**2,
    #                   projection_func_coef[17][1]*x**2*y**0,
    #                   projection_func_coef[18][1]*x**2*y**1,
    #                   projection_func_coef[19][1]*x**3*y**0
    #                   ])
    # check_projection_func = {'img2phys_x':img2phys_x,
    #                          'img2phys_y':img2phys_y,
    #                          'phys2img_x':phys2img_x,
    #                          'phys2img_y':phys2img_y}
    
    # print('check progection function :')
    # i = 0
    # for key, value in check_projection_func.items():
    #     print('{} = {} {}'.format(key, value, value == new_coordinate[i]))
    #     i += 1