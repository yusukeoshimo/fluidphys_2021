#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021-09-29 18:04:44
# MC_dropout.py

import sys
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model

# 参照：https://st1990.hatenablog.com/entry/2019/07/31/010010
class MontecarloDropout:
    def __init__(self):
        return

    def build_model(self, model_file_path):
        """
        keras modelからMontecarloDropoutに対応したモデルを作成
        build monte carlo dropout model base on keras_model.
        """

        model = self.__load_model(model_file_path)

        # change dropout layer to dropout layer that can use dropout in inference.
        # ドロップアウト層を推論時にもドロップアウトできるドロップアウト層に変更する。
        for ily, layer in enumerate(model.layers):
            # input layer
            if ily == 0:
                input = layer.input
                h = input
            # is dropout layer ?
            if 'dropout' in layer.name:
                # change dropout layer
                h = Dropout(layer.rate)(h, training=True)
            else:
                h = layer(h)

        self.model = Model(input, h)
        return

    def md_predict(self, xs, sampling_num):
        """
        predict with using monte carlo dropout sampling.
        return prediction average, std

        xs : input sample array. xs = x0, x1, x2, ...
        """
        pre_ys = []
        for ismp in range(sampling_num):
            pre_y = self.model.predict(xs)
            pre_ys.append(pre_y)
        pre_ys = np.array(pre_ys)

        # calculate ave, std
        pre_ave = np.average(pre_ys, axis=0)
        pre_std = np.std(pre_ys, axis=0)

        return pre_ave, pre_std

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(u'使い方: python {}'.format(os.path.basename(sys.argv[0])) +
            ' cnn_model\n')
        sys.exit(0) # 正常終了, https://www.sejuku.net/blog/24331
    mcd = MontecarloDropout
    mcd.build_model(sys.argv[1])