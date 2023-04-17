import matplotlib.pyplot as plt
import wfdb
from biosppy.signals.tools import filter_signal
from keras.engine.saving import model_from_json


def load_data(sample_path):
    '''
    使用wfdb读取数据
    '''
    sig, fields = wfdb.rdsamp(sample_path)  # 返回信号值
    length = len(sig)  # 获取信号长度
    fs = fields['fs']  # 读取采样频率

    return sig, length, fs


if __name__ == '__main__':
    sample_path = r'G:\2-pycharm_file\1-python_programe\2-面试算法题\生理信号挑战赛\data\training_1\data_0_1'

    # 加载QRS波模型和房颤检测CNN模型
    qrs = model_from_json(open('./QRS_detector/models/CNN.json').read())  # 加载QRS波CNN预测模型
    qrs.load_weights('./QRS_detector/models/CNN.h5')
    print('\nQRS定位CNN模型结构\n')
    qrs.summary()
    model = model_from_json(open('deep.json').read())
    model.load_weights('deep.h5')# 房颤检测CNN预测模型
    print('\n房颤检测CNN模型结构\n')
    model.summary()

    sig, length, fs = load_data(sample_path)  # 加载心电数据
    print(f'信号的采样频率：{fs}hz')
    print(f'信号的长度：{length}，{length / 500}')
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
    print(f'裁剪为:{n_seg}')
