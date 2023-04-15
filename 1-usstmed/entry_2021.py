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
from keras.models import Sequential,Model,load_model,model_from_json

def load_data(sample_path):
    sig, fields = wfdb.rdsamp(sample_path)
    length = len(sig)
    fs = fields['fs']

    return sig, length, fs

def challenge_entry(sample_path):
    """
    This is a baseline method.
    """

    sig, length, fs = load_data(sample_path)
    sig[:,0] = filter_signal(sig[:,0],ftype='FIR',band='bandpass',order=50,frequency=[0.5,45],sampling_rate=fs)[0]
    sig[:,1] = filter_signal(sig[:,1],ftype='FIR',band='bandpass',order=50,frequency=[0.5,45],sampling_rate=fs)[0]
    n_seg = (length-800)//1600
    qrs_pred = []
    af_pred = []
    if length<4000:
        sig[:,0] -= np.mean(sig[:,0])
        sig[:,1] -= np.mean(sig[:,1])
        ecg0 = resample(sig[:,0],int(len(sig[:,0])*2.5))
        ecg1 = resample(sig[:,1],int(len(sig[:,1])*2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0),np.expand_dims(ecg1, 0)],axis=0)
        qrs_pred = qrs.predict(np.expand_dims(ecg, -1))[1,:,0]
        af_pred = model.predict(np.expand_dims(sig, 0))[0,:,0]
    else:
        for s in range(n_seg-1):
            temp = sig[1600*s:1600*s+2400,:].copy()
            temp[:,0] = temp[:,0]-temp[:,0].mean()
            temp[:,1] = temp[:,1]-temp[:,1].mean()
            ecg0 = resample(temp[:,0],int(len(temp[:,0])*2.5))
            ecg1 = resample(temp[:,1],int(len(temp[:,1])*2.5))
            ecg = np.concatenate([np.expand_dims(ecg0, 0),np.expand_dims(ecg1, 0)],axis=0)
            fr = 125 if s>0 else 0
            qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[1,:,0][fr:-125]))
            af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0,:,0][fr//5:-25]))
        temp = sig[1600*n_seg-1600:,:].copy()
        temp[:,0] = temp[:,0]-temp[:,0].mean()
        temp[:,1] = temp[:,1]-temp[:,1].mean()
        ecg0 = resample(temp[:,0],int(len(temp[:,0])*2.5))
        ecg1 = resample(temp[:,1],int(len(temp[:,1])*2.5))
        ecg = np.concatenate([np.expand_dims(ecg0, 0),np.expand_dims(ecg1, 0)],axis=0)
        qrs_pred.extend(list(qrs.predict(np.expand_dims(ecg, -1))[:,:,0][125:]))
        af_pred.extend(list(model.predict(np.expand_dims(temp, 0))[0,:,0][25:]))

    rs = QRS_decision(np.array(qrs_pred))
    rs = rs//2.5
    res = np.array(af_pred)
    pred_af = np.where(res[(rs//16).astype(int)]>0.52)[0]
    change = np.where(np.diff(pred_af)>4)[0]
    end_points = []
    start = []
    end = []
    good_af = []
    i=0
    while i<len(pred_af):
        if i+4<len(pred_af) and pred_af[i]+4==pred_af[i+4]:
            n=4
            while i+n<len(pred_af) and pred_af[i]+n==pred_af[i+n]:
                n+=1
            while i+n<len(pred_af) and pred_af[i+n]-pred_af[i+n-1]<4:
                n+=1
            good_af.extend(pred_af[i:i+n])
            i+=n
        else:
            i+=1
    if len(good_af)<6:
        end_points = []
    else:
        change = np.where(np.diff(good_af)>5)[0]
        if len(change)==0:
            if good_af[0]<2:
                start.append(0)
            else:
                start.append(rs[good_af[0]]-20)
            if good_af[-1]>=len(rs)-1:
                end.append(length-1)
            else:
                end.append(rs[good_af[-1]]+20)
        else:
            if good_af[0]<2:
                start.append(0)
            else:
                start.append(rs[good_af[0]]-20)
            end.append(rs[good_af[change[0]]]+20)
            for c in range(len(change)):
                start.append(rs[good_af[change[c]+1]]-20)
                if c<len(change)-1:
                    end.append(rs[good_af[change[c+1]]]+20)
                else:
                    end.append(rs[good_af[-1]]+20)
            if good_af[-1]>=len(rs)-1:
                end[-1] = length-1
                
        if end[-1]>length-1:
            end[-1] = length-1
        if start[0]<0:
            start[0] = 0

    start = np.expand_dims(start, -1)
    end = np.expand_dims(end, -1)
    start_end = np.concatenate((start, end), axis=-1).tolist()
    end_points.extend(start_end)       
    pred_dcit = {'predict_endpoints': end_points}
    
    return pred_dcit

if __name__ == '__main__':
    DATA_PATH = r'G:\2-pycharm_file\1-python_programe\2-面试算法题\生理信号挑战赛\data'+'\\training_1'
    RESULT_PATH = './result/'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    test_set = open(os.path.join(DATA_PATH, 'RECORDS'), 'r').read().splitlines()
    qrs = model_from_json(open('./QRS_detector/models/CNN.json').read())
    qrs.load_weights('./QRS_detector/models/CNN.h5')
    model = model_from_json(open('deep.json').read())
    model.load_weights('deep.h5')
    model.summary()
    for i, sample in enumerate(test_set):
        print(sample)
        sample_path = os.path.join(DATA_PATH, sample)
        pred_dict = challenge_entry(sample_path)
        save_dict(os.path.join(RESULT_PATH, sample+'.json'), pred_dict)
