import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import wfdb
from biosppy.signals.tools import filter_signal
from keras.engine.saving import model_from_json
from keras.utils import plot_model
from scipy.signal import resample
from QRS_detector.decision import *


def load_data(sample_path):
    '''
    使用wfdb读取数据
    '''
    sig, fields = wfdb.rdsamp(sample_path)  # 返回信号值
    length = len(sig)  # 获取信号长度
    fs = fields['fs']  # 读取采样频率

    return sig, length, fs


def plot_image(data):
    plt.plot(data)
    plt.show()


def svae_mode_txt(model, name):
    '''
    将模型文件保存为txt
    '''
    txt_path = './model_txt/' + f'{name}.txt'
    with open(txt_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(name + "保存成功")


if __name__ == '__main__':
    sample_path = r'G:\2-pycharm_file\1-python_programe\2-面试算法题\生理信号挑战赛\data\training_1\data_3_1'

    # # 加载QRS波模型和房颤检测CNN模型
    qrs = model_from_json(open('./QRS_detector/models/CNN.json').read())  # 加载QRS波CNN预测模型
    qrs.load_weights('./QRS_detector/models/CNN.h5')
    print('\nQRS定位CNN模型结构\n')
    qrs.summary()
    # svae_mode_txt(qrs, 'QRS检测模型')
    model = model_from_json(open('deep.json').read())
    model.load_weights('deep.h5')  # 房颤检测CNN预测模型
    print('\n房颤检测CNN模型结构\n')
    model.summary()
    # svae_mode_txt(model, '房颤检测模型')
    sig, length, fs = load_data(sample_path)  # 加载心电数据
    print(f'信号的采样频率：{fs}hz')
    print(f'信号的长度：{length}，{length / 200}')
    plt.figure(num=1, dpi=600)
    plt.subplot(211)
    plt.plot(sig[:1000, 0], c='b', label='channel_1')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(sig[:1000, 1], c='r', label='channel_2')
    plt.suptitle("Original signal")
    plt.legend(loc='upper right')
    plt.tight_layout(2)
    plt.show()

    sig[:, 0] = filter_signal(sig[:, 0], ftype='FIR', band='bandpass', order=50, frequency=[0.5, 45], sampling_rate=fs)[
        0]
    sig[:, 1] = filter_signal(sig[:, 1], ftype='FIR', band='bandpass', order=50, frequency=[0.5, 45], sampling_rate=fs)[
        0]
    plt.figure(num=2, dpi=600)
    plt.subplot(211)
    plt.plot(sig[:1000, 0], c='b', label='channel_1')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(sig[:1000, 1], c='r', label='channel_2')
    plt.suptitle("Deionised signal")
    plt.legend(loc='upper right')
    plt.tight_layout(2)
    plt.show()
    n_seg = (length - 800) // 1600
    print(f'裁剪为:{n_seg / 200}s')
    qrs_pred = []  # QRS预测
    af_pred = []  # 房颤预测
    if length < 4000:  # 4000/200=20s
        sig[:, 0] -= np.mean(sig[:, 0])  # 减去信号的均值，去除直流分量
        sig[:, 1] -= np.mean(sig[:, 1])
        # 重采样至500hz，使用傅里叶方法沿给定轴对“x”到“num”样本进行重采样。重新采样的信号从与“x”相同的值开始，但采样的间距为“len（x）num（x）间距”。
        ecg0 = resample(sig[:, 0], int(len(sig[:, 0]) * 2.5))
        ecg1 = resample(sig[:, 1], int(len(sig[:, 1]) * 2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)
        qrs_pred = qrs.predict(np.expand_dims(ecg, -1))[1, :, 0]
        af_pred = model.predict(np.expand_dims(sig, 0))[0, :, 0]
    else:
        for s in range(n_seg - 1):
            temp = sig[1600 * s:1600 * s + 2400, :].copy()
            temp[:, 0] = temp[:, 0] - temp[:, 0].mean()
            temp[:, 1] = temp[:, 1] - temp[:, 1].mean()
            ecg0 = resample(temp[:, 0], int(len(temp[:, 0]) * 2.5))
            ecg1 = resample(temp[:, 1], int(len(temp[:, 1]) * 2.5))
            # plot_image(ecg0)
            ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)
            fr = 125 if s > 0 else 0
            ecg_1=np.expand_dims(ecg,-1)
            predict_test=qrs.predict(ecg_1)
            np.save("./test_npy/ecg_1",ecg_1)
            np.save('./test_npy/qrs_predict',predict_test)
            qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[1, :, 0][fr:-125]))
            af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0, :, 0][fr // 5:-25]))
        temp = sig[1600 * n_seg - 1600:, :].copy()
        temp[:, 0] = temp[:, 0] - temp[:, 0].mean()
        temp[:, 1] = temp[:, 1] - temp[:, 1].mean()
        ecg0 = resample(temp[:, 0], int(len(temp[:, 0]) * 2.5))
        ecg1 = resample(temp[:, 1], int(len(temp[:, 1]) * 2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0), np.expand_dims(ecg1, 0)], axis=0)
        qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[:, :, 0][125:]))
        af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0, :, 0][25:])) #把值放到af_pred末尾

    rs = QRS_decision(np.array(qrs_pred))  # 根据r峰值的预测结果判断QRS波群位置
    rs = rs // 2.5  # 将QRS波群位置还原到原始采样率
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
    print(end_points)
