# STWave
This is a PyTorch implementation of the paper: [When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks](https://ieeexplore.ieee.org/document/10184591).

[08/2022] STWave is accepted by ICDE 2023!

![The architecture of the proposed STWave. FC: fully-connected layer, DWT: discrete wavelet transform.](https://github.com/LMissher/STWave/blob/main/stwave.png)

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt.

The Pytorch_Wavelets package should be manually installed according to this [github page](https://github.com/fbcotter/pytorch_wavelets)

## Data Preparation

### PeMSD3 & PeMSD4 & PeMSD7 & PeMSD8
download the data [PeMSD*](https://pan.baidu.com/share/init?surl=ZPIiOM__r1TRlmY4YGlolw) with code: p72z. 

### PeMSD7(M) & PeMSD7(L)
download the data [PeMSD7(M)](https://github.com/VeritasYin/STGCN_IJCAI-18/tree/master/data_loader).
email authors of STGCN to get the data PeMSD7(L).

### Before Training
make folders of cpt and log.

## Model Training

# PeMSD3
```
python train.py --config ./config/PeMSD3.conf
```

# PeMSD4
```
python train.py --config ./config/PeMSD4.conf
```

# PeMSD7
```
python train.py --config ./config/PeMSD7.conf
```

# PeMSD8
```
python train.py --config ./config/PeMSD8.conf
```

# PeMSD7(M)
```
python train.py --config ./config/PeMSD7(M).conf
```

# PeMSD7(L)
```
python train.py --config ./config/PeMSD7(L).conf
```
