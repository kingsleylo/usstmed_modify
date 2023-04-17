#!/usr/bin/env python3
import numpy as np
import os
import sys
import wfdb
from utils import qrs_detect, comp_cosEn, save_dict
from scipy.signal import resample
from biosppy.signals.tools import filter_signal
from QRS_detector.decision import *
from keras import backend as K
from keras import layers
from keras.layers import *
from keras import Input
from keras.optimizers import *
from keras.models import Sequential, Model, load_model, model_from_json


def load_data(sample_path):
    '''
    使用wfdb读取数据
    '''
    sig, fields = wfdb.rdsamp(sample_path)  # 返回信号值
    length = len(sig)  # 获取信号长度
    fs = fields['fs']  # 读取采样频率

    return sig, length, fs


def challenge_entry(sample_path):
    '''
    :sample_path:样本的数据文件
    函数的作用：预测QRS波群位置和房颤的检测。
    流程：首先通过load_data函数加载样本数据，并对数据进行滤波处理。
    然后根据数据长度判断分段情况。如果数据长度小于4000，则将信号重采样，然后预测QRS波群位置和房颤。
    如果数据长度大于4000，则对信号进行分段处理，每段长度为1600，分别预测QRS波群位置和房颤。
    最后，根据预测结果判断房颤的存在，并返回结果。
    '''

    sig, length, fs = load_data(sample_path)  # 加载心电数据
    # 使用带通滤波进行滤波，保留0.5~45Hz之间的信号，去除噪声
    sig[:, 0] = filter_signal(sig[:, 0], ftype='FIR', band='bandpass', order=50, frequency=[0.5, 45], sampling_rate=fs)[
        0]
    sig[:, 1] = filter_signal(sig[:, 1], ftype='FIR', band='bandpass', order=50, frequency=[0.5, 45], sampling_rate=fs)[
        0]
    n_seg = (length - 800) // 1600  # 将数据统一裁剪成8s的长度，(8*0.5)4s为重叠采样,裁剪的片段个数
    qrs_pred = []  # QRS预测
    af_pred = []  # 房颤预测
    if length < 4000:  # 4000/200=20s
        sig[:, 0] -= np.mean(sig[:, 0])  # 减去信号的均值，去除直流分量
        sig[:, 1] -= np.mean(sig[:, 1])
        # 重采样至500hz，使用快速傅里叶方法沿给定轴对“x”到“num”样本进行重采样。重新采样的信号从与“x”相同的值开始，但采样的间距为“len（x）num（x）间距”。
        ecg0 = resample(sig[:, 0], int(len(sig[:, 0]) * 2.5))
        ecg1 = resample(sig[:, 1], int(len(sig[:, 1]) * 2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)  # 合并两个通道的信号
        qrs_pred = qrs.predict(np.expand_dims(ecg, -1))[1, :, 0]
        af_pred = model.predict(np.expand_dims(sig, 0))[0, :, 0]
    else:
        for s in range(n_seg - 1):
            temp = sig[1600 * s:1600 * s + 2400, :].copy()  # 原始信号裁剪成长度为2400的小段，2400/200=12S
            temp[:, 0] = temp[:, 0] - temp[:, 0].mean()
            temp[:, 1] = temp[:, 1] - temp[:, 1].mean()
            ecg0 = resample(temp[:, 0], int(len(temp[:, 0]) * 2.5))  # 将信号进行重采样，使其采样率变为500Hz，样本点变为6000
            ecg1 = resample(temp[:, 1], int(len(temp[:, 1]) * 2.5))
            ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)
            fr = 125 if s > 0 else 0
            qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[1, :, 0][fr:-125]))
            af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0, :, 0][fr // 5:-25]))
        temp = sig[1600 * n_seg - 1600:, :].copy()
        temp[:, 0] = temp[:, 0] - temp[:, 0].mean()
        temp[:, 1] = temp[:, 1] - temp[:, 1].mean()
        ecg0 = resample(temp[:, 0], int(len(temp[:, 0]) * 2.5))
        ecg1 = resample(temp[:, 1], int(len(temp[:, 1]) * 2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)
        qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[:, :, 0][125:]))
        af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0, :, 0][25:]))

    rs = QRS_decision(np.array(qrs_pred))  # 根据r峰值的预测结果判断QRS波群位置
    rs = rs // 2.5 # 将QRS波群位置还原到原始采样率
    res = np.array(af_pred)
    pred_af = np.where(res[(rs // 16).astype(int)] > 0.52)[0]
    change = np.where(np.diff(pred_af) > 4)[0]
    end_points = []
    start = []
    end = []
    good_af = []
    i = 0
    while i < len(pred_af):
        if i + 4 < len(pred_af) and pred_af[i] + 4 == pred_af[i + 4]:
            n = 4
            while i + n < len(pred_af) and pred_af[i] + n == pred_af[i + n]:
                n += 1
            while i + n < len(pred_af) and pred_af[i + n] - pred_af[i + n - 1] < 4:
                n += 1
            good_af.extend(pred_af[i:i + n])
            i += n
        else:
            i += 1
    if len(good_af) < 6:
        end_points = []
    else:
        change = np.where(np.diff(good_af) > 5)[0]
        if len(change) == 0:
            if good_af[0] < 2:
                start.append(0)
            else:
                start.append(rs[good_af[0]] - 20)
            if good_af[-1] >= len(rs) - 1:
                end.append(length - 1)
            else:
                end.append(rs[good_af[-1]] + 20)
        else:
            if good_af[0] < 2:
                start.append(0)
            else:
                start.append(rs[good_af[0]] - 20)
            end.append(rs[good_af[change[0]]] + 20)
            for c in range(len(change)):
                start.append(rs[good_af[change[c] + 1]] - 20)
                if c < len(change) - 1:
                    end.append(rs[good_af[change[c + 1]]] + 20)
                else:
                    end.append(rs[good_af[-1]] + 20)
            if good_af[-1] >= len(rs) - 1:
                end[-1] = length - 1

        if end[-1] > length - 1:
            end[-1] = length - 1
        if start[0] < 0:
            start[0] = 0

    start = np.expand_dims(start, -1)
    end = np.expand_dims(end, -1)
    start_end = np.concatenate((start, end), axis=-1).tolist()
    end_points.extend(start_end)
    pred_dcit = {'predict_endpoints': end_points}

    return pred_dcit


if __name__ == '__main__':
    DATA_PATH = r'G:\2-pycharm_file\1-python_programe\2-面试算法题\生理信号挑战赛\data' + '\\training_1'
    RESULT_PATH = './result/'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    qrs = model_from_json(open('./QRS_detector/models/CNN.json').read())  # 加载QRS波CNN预测模型
    qrs.load_weights('./QRS_detector/models/CNN.h5')
    model = model_from_json(open('deep.json').read())
    model.load_weights('deep.h5')  # 房颤检测CNN预测模型
    model.summary()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)
        save_dict(os.path.join(RESULT_PATH, sample + '.json'), pred_dict)
