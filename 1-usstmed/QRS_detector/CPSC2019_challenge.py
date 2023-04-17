import numpy as np


def CPSC2019_challenge(result):
    '''
    根据心电图（ECG）的结果计算心率（HR）和QRS复合波的位置.
    '''
    pos = np.argwhere(result > 0.5).flatten()  # 找到结果变量中结果大于0.5的位置并展开成一维数组（QRS波幅值应该大于0.5.）
    rpos = []
    pre = 0  # 前一个位置
    last = len(pos)  # 后一个位置
    for j in np.where(np.diff(pos) > 2)[0]:  # 找到相邻位置之间间隔大于2的位置
        if j - pre > 2:
            rpos.append((pos[pre] + pos[j]) * 4)  # 记录r峰的位置
        pre = j + 1
    rpos.append((pos[pre] + pos[last - 1]) * 4)  # QRS波区间末尾
    qrs = np.array(rpos)
    qrs_diff = np.diff(qrs)
    check = True
    while check:
        qrs_diff = np.diff(qrs)  # 计算r间期
        for r in range(len(qrs_diff)):  # 如果r间期小于100，那么会检查这两个位置中哪个位置的值更大，如果前一个位置的值更大，
            # 那么就删除后一个位置，否则删除前一个位置
            if qrs_diff[r] < 100:
                if result[int(qrs[r] / 8)] > result[int(qrs[r + 1] / 8)]:
                    qrs = np.delete(qrs, r + 1)
                    check = True
                    break
                else:
                    qrs = np.delete(qrs, r)
                    check = True
                    break
            check = False
    hr = np.array([loc for loc in qrs if (loc > 2750 and loc < 4750)])  # 筛选出2750到4750的区间
    if len(hr) > 1:
        hr = round(60 * 500 / np.mean(np.diff(hr)))  # 计算心率
    else:
        hr = 80
    return hr, qrs
