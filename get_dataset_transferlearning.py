import torch
import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split
import random

def Source_TrainDataset(snr):
    # data_path = "/data/fuxue/Dataset_LFM/TrainData/0-5 Radar/X_train_snr=" + str(snr) + "dB.mat"
    data_path = "/data/ylj/PycharmProjects/transferlearning/LhwDataset/16psk/IQ_16psk2000_SNR=" + str(snr) + ".mat"
    data = scio.loadmat(data_path)
    x = data.get('IQ')
    x = x.transpose(0, 2, 1)
    """
        label_path = "/data/fuxue/Dataset_LFM/TrainData/0-5 Radar/Y_train_snr=" + str(snr) + "dB.mat"
        data = scio.loadmat(label_path)
        y = data.get('radar_labels')
        y = y.astype(np.uint8)
    """
    y = np.zeros((12000, 1))
    for i in range(6):
        y[2000*i: 2000*i+2000] = i

    X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.1, random_state=30)
    return  X_train, X_val, Y_train, Y_val

def get_class_index(targets, one_class):
    return [index for (index, value) in enumerate(targets) if value == one_class]

def Target_TrainDataset(snr, k):
    num = 6 #类别总数
    # data_path = "/data/fuxue/Dataset_LFM/TrainData/6-11 Radar/X_train_snr=" + str(snr) + "dB.mat"
    data_path = "/data/ylj/PycharmProjects/transferlearning/LhwDataset/bpsk/IQ_bpsk2000_SNR=" + str(snr) + ".mat"
    data = scio.loadmat(data_path)
    X_train = data.get('IQ')
    X_train = X_train.transpose(0, 2, 1)
    # label_path = "/data/fuxue/Dataset_LFM/TrainData/6-11 Radar/Y_train_snr=" + str(snr) + "dB.mat"
    # data = scio.loadmat(label_path)
    # Y_train = data.get('radar_labels')-6
    # Y_train = Y_train.astype(np.uint8)
    Y_train = np.zeros((12000, 1))
    for i in range(num):
        Y_train[2000*i: 2000*i+2000] = i
    X_train_K_Shot = np.zeros([int(k*num), 2, 6000])
    Y_train_K_Shot = np.zeros([int(k*num), 1]).astype(np.uint8)

    for i in range(num):
        index_shot = get_class_index(Y_train, i)
        random_shot = random.sample(index_shot, k)

        X_train_K_Shot[i*k:i*k+k,:,:] = X_train[random_shot,:,:]
        Y_train_K_Shot[i*k:i*k+k] = Y_train[random_shot]

    X_train_K_Shot, X_val, Y_train_K_Shot, Y_val = train_test_split(X_train_K_Shot, Y_train_K_Shot, test_size=0.3, random_state=30)

    return X_train_K_Shot, X_val, Y_train_K_Shot, Y_val

def TestDataset(snr):
    # data_path = "/data/fuxue/Dataset_LFM/TestData/6-11 Radar/X_test_snr=" + str(snr) + "dB.mat"
    data_path = "/data/ylj/PycharmProjects/transferlearning/LhwDataset/bpsk/IQ_bpsk1000_test_SNR=" + str(snr) + ".mat"
    data = scio.loadmat(data_path)
    x = data.get('IQ')
    x = x.transpose(0, 2, 1)
    y = np.zeros((6000, 1))
    for i in range(6):
        y[1000*i: 1000*i+1000] = i


    # label_path = "/data/fuxue/Dataset_LFM/TestData/6-11 Radar/Y_test_snr=" + str(snr) + "dB.mat"
    # data = scio.loadmat(label_path)
    # y = data.get('radar_labels')-6
    # y = y.astype(np.uint8)
    return x, y

