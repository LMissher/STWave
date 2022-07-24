import logging
import numpy as np
import os
import pickle
import sys
import torch
import math
from pytorch_wavelets import DWT1DForward, DWT1DInverse

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def _compute_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def disentangle(data, w, j):
    # Disentangle
    dwt = DWT1DForward(wave=w, J=j)
    idwt = DWT1DInverse(wave=w)
    torch_traffic = torch.from_numpy(data).transpose(1,-1).reshape(data.shape[0]*data.shape[2], -1).unsqueeze(1)
    torch_trafficl, torch_traffich = dwt(torch_traffic.float())
    placeholderh = torch.zeros(torch_trafficl.shape)
    placeholderl = []
    for i in range(j):
        placeholderl.append(torch.zeros(torch_traffich[i].shape))
    torch_trafficl = idwt((torch_trafficl, placeholderl)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    torch_traffich = idwt((placeholderh, torch_traffich)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    trafficl = torch_trafficl.numpy()
    traffich = torch_traffich.numpy()
    return trafficl, traffich

def loadData(args):
    # Traffic
    Traffic = np.squeeze(np.load(args.traffic_file)['data'], -1)
    print(Traffic.shape)
    # train/val/test 
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # X, Y
    trainX, trainY = seq2instance(train, args.T1, args.T2)
    valX, valY = seq2instance(val, args.T1, args.T2)
    testX, testY = seq2instance(test, args.T1, args.T2)
    # disentangling
    trainXL, trainXH = disentangle(trainX, args.w, args.j)
    trainYL, trainYH = disentangle(trainY, args.w, args.j)
    valXL, valXH = disentangle(valX, args.w, args.j)
    testXL, testXH = disentangle(testX, args.w, args.j)
    # normalization
    mean, std = np.mean(trainX), np.std(trainX)
    trainXL, trainXH = (trainXL - mean) / std, (trainXH - mean) / std
    valXL, valXH = (valXL - mean) / std, (valXH - mean) / std
    testXL, testXH = (testXL - mean) / std, (testXH - mean) / std
    trainX, valX, testX = (trainX - mean) / std, (valX - mean) / std, (testX - mean) / std
    # temporal embedding
    tmp = {'PeMSD3':6,'PeMSD4':1,'PeMSD7':1,'PeMSD8':5, 'PeMSD7L':2, 'PeMSD7M':2}
    days = {'PeMSD3':7,'PeMSD4':7,'PeMSD7':7,'PeMSD8':7, 'PeMSD7L':5, 'PeMSD7M':5}
    TE = np.zeros([num_step, 2])
    startd = (tmp[args.Dataset] - 1) * 288
    df = days[args.Dataset]
    startt = 0
    for i in range(num_step):
        TE[i,0] = startd //  288
        startd = (startd + 1) % (df * 288)
        TE[i,1] = startt
        startt = (startt + 1) % 288
    # train/val/test
    train = TE[: train_steps]
    val = TE[train_steps : train_steps + val_steps]
    test = TE[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.T1, args.T2)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.T1, args.T2)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.T1, args.T2)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
    
    return trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, mean, std