import sys

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import resample

if __name__ == "__main__":
    path = './1-usstmed/test_npy'

    ecg = np.load(path + './ecg_1.npy')
    qrs = np.load(path+'./qrs_predict.npy')
    print(ecg.shape)
    print(qrs.shape)
    print(qrs)
    qrs_location=list(qrs[1, :, 0])
    q1=qrs_location[125:-125]
    plt.plot(qrs_location)
    plt.show()
    print(qrs_location)
    print(len(qrs_location))

    print(len(q1))