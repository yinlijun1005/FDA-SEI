import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
import scipy.io as scio

def TrainDataset(num, snr):
    num = 6
    data_path = "/data/ylj/PycharmProjects/transferlearning/LhwDataset/16psk/IQ_16psk2000_SNR=" + str(snr) + ".mat"
    data = scio.loadmat(data_path)
    x = data.get('IQ')
    # x = x.transpose(0, 2, 1)

    y = np.zeros((12000, 1))
    for i in range(6):
        y[2000 * i: 2000 * i + 2000] = i

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.3, random_state=30)
    return X_train, X_val, Y_train, Y_val


def TestDataset(snr):
    data_path = "/data/ylj/PycharmProjects/transferlearning/LhwDataset/16psk/IQ_16psk1000_test_SNR=" + str(snr) + ".mat"
    data = scio.loadmat(data_path)
    x = data.get('IQ')
    # x = x.transpose(0, 2, 1)
    y = np.zeros((6000, 1))
    for i in range(6):
        y[1000*i: 1000*i+1000] = i

    return x, y
