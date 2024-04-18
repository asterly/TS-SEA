
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch

def load_base(data_path,mode="train"):
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    return train_X,train_y,test_X,test_y

def load_HAR(mode="train"):
    data_path = "/workspace/CA-TCC/data/HAR/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels
    print(train_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_EEG(mode="train"):
    data_path = "/workspace/CA-TCC/data/SleepEDF/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
        
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    # val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    # test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']
    print(train.shape)
    print(val.shape)
    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    print("train_X.shape",train_X.shape)
    print("test_X.shape",test_X.shape)
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) # [N, L, C]
    # train_X = np.transpose(train_X,(0,2,1))
    # test_X = np.transpose(test_X,(0,2,1))
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_Epi(mode="train"):
    data_path = "/workspace/CA-TCC/data/Epilepsy/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_Waveform():
    data_path = "./Waveform/"
    train_ = torch.load(data_path + "train.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    test = test_['samples']
    # test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train
    train_y = train_labels
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_HAR_fft(mode="train"):
    data_path = "/workspace/CA-TCC/data/HAR/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples'] # [5881, 9, 128]
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_HAR_seasonal(mode="train"):
    data_path = "/workspace/CA-TCC/data/HAR/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples'] # [5881, 9, 128]
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
    test_X = test
    test_y = test_labels

    train_X = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            #print("seasonal",result.seasonal)
            trend = pd.Series(result.trend)
            trend = trend.ffill().bfill()
            trend = trend.to_numpy()
            # print("trend",trend)
            #print("resid",result.resid.fillna(method='ffill').fillna(method='bfill'))
            dims_list.append(trend)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X = torch.Tensor(sean_list)
    print(train_X.shape)
    train_X = torch.permute(train_X,(0,2,1))
    print(train_X.shape)

    test_X = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            trend = pd.Series(result.trend)
            trend = trend.ffill().bfill()
            trend = trend.to_numpy()
            dims_list.append(trend)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X = torch.Tensor(sean_list)
    print(test_X.shape)
    test_X = torch.permute(test_X,(0,2,1))
    print(test_X.shape)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_HAR_val():
    data_path = "/workspace/CA-TCC/data/HAR/"
    # train_ = torch.load(data_path + "val.pt")
    val_ = torch.load(data_path + "val.pt")

    val = val_['samples']
    val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']

        
    val_X = val
    val_y = val_labels
    print(val_X.shape)
    scaler = StandardScaler()
    scaler.fit(val_X.reshape(-1, val_X.shape[-1]))
    val_X = scaler.transform(val_X.reshape(-1, val_X.shape[-1])).reshape(val_X.shape)

    labels = np.unique(val_y)
    transform = {k: i for i, k in enumerate(labels)}
    val_y = np.vectorize(transform.get)(val_y)
    return val_X, val_y


def load_EEG_fft(mode="train"):
    data_path = "/workspace/CA-TCC/data/SleepEDF/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    # val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    # test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # train_X = np.transpose(train_X,(0,2,1))
    # test_X = np.transpose(test_X,(0,2,1))
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()

def load_EEG_seasonal(mode="train"):
    data_path = "/workspace/CA-TCC/data/SleepEDF/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}_sea.pt")
    else:
        train_ = torch.load(data_path + "train_sea.pt")
        
    val_ = torch.load(data_path + "val_sea.pt")
    test_ = torch.load(data_path + "test_sea.pt")
    train = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    val = val_['samples']
    # val = torch.transpose(val, 1, 2)
    val_labels = val_['labels']
    test = test_['samples']
    # test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']
    print(train.shape)
    print(val.shape)
    train_X = torch.cat([train, val])
    train_y = torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    print("train_X.shape",train_X.shape)
    print("test_X.shape",test_X.shape)
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y) # [N, L, C]
    # train_X = np.transpose(train_X,(0,2,1))
    # test_X = np.transpose(test_X,(0,2,1))
    return train_X, train_y, test_X, test_y
    # return torch.cat([train, val]).numpy(), torch.cat([train_labels, val_labels]).numpy(), test.numpy(), test_labels.numpy()


def load_Epi_fft(mode="train"):
    data_path = "/workspace/CA-TCC/data/Epilepsy/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_Epi_seasonal(mode="train"):
    data_path = "/workspace/CA-TCC/data/Epilepsy/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    #val_ = torch.load(data_path + "val.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    #val = val_['samples']
    #val = torch.transpose(val, 1, 2)
    #val_labels = val_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train #torch.cat([train, val])
    train_y = train_labels #torch.cat([train_labels, val_labels])
        
    test_X = test
    test_y = test_labels
    # print(train_X.shape)
    # train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    # test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    # train_X = train_X_fft.transpose(1, 2)
    # test_X = test_X_fft.transpose(1, 2)
    print(train_X.shape)
    seasonal_list = []
    for sample in train_X :
        result = seasonal_decompose(sample, model='additive', period=30)
        seasonal_list.append(result.seasonal)

    print(len(seasonal_list))
    train_X = np.stack(seasonal_list)
    train_X = torch.Tensor(train_X)
    train_X = train_X.reshape(train_X.shape[0],-1,1)
    print(train_X.shape)
    
    seasonal_list = []
    for sample in test_X :
        result = seasonal_decompose(sample, model='additive', period=30)
        seasonal_list.append(result.seasonal)
        
    print(len(seasonal_list))
    test_X = np.stack(seasonal_list)
    test_X = torch.Tensor(test_X)
    test_X = test_X.reshape(test_X.shape[0],-1,1)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def load_Waveform_fft():
    data_path = "./data/Waveform/"
    train_ = torch.load(data_path + "train.pt")
    test_ = torch.load(data_path + "test.pt")
    train = train_['samples']
    train = torch.transpose(train, 1, 2)
    train_labels = train_['labels']
    test = test_['samples']
    test = torch.transpose(test, 1, 2)
    test_labels = test_['labels']

    train_X = train
    train_y = train_labels
    test_X = test
    test_y = test_labels

    train_X_fft = torch.fft.fft(train_X).abs()
    test_X_fft = torch.fft.fft(test_X).abs()
    train_X = train_X_fft.transpose(1, 2)
    test_X = test_X_fft.transpose(1, 2)


    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y

def load_HAR_two_view(mode="train"):
    train_X, train_y, test_X, test_y = load_HAR(mode)
    train_X_fft, _, test_X_fft, _ = load_HAR_fft(mode)
    train_X_sea, _, test_X_sea, _ = load_HAR_seasonal(mode)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y
def load_HAR_two_view_for_tf(mode="train"):
    train_X, train_y, test_X, test_y = load_HAR(mode)
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft(mode)
    train_X_sea, _, test_X_sea, _ = load_HAR_seasonal(mode)
    return [train_X, train_X_sea], train_y, [test_X, test_X_sea], test_y
def load_EEG_two_view(mode="train"):
    train_X, train_y, test_X, test_y = load_EEG(mode)
    train_X_fft, _, test_X_fft, _ = load_EEG_fft(mode)
    train_X_sea, _, test_X_sea, _ = load_EEG_seasonal(mode)

    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y

def load_Epi_two_view(mode="train"):
    train_X, train_y, test_X, test_y = load_Epi(mode) # train_X: 9200, 178, 1, train_y: 9200,
    train_X_fft, _, test_X_fft, _ = load_Epi_fft(mode)
    train_X_sea, _, test_X_sea, _ = load_Epi_seasonal(mode)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y

def load_Waveform_two_view():
    train_X, train_y, test_X, test_y = load_Waveform() # train_X: 9200, 178, 1, train_y: 9200,
    train_X_fft, _, test_X_fft, _ = load_Waveform_fft()
    return [train_X, train_X_fft], train_y, [test_X, test_X_fft], test_y



def load_isruc_two_view(mode="train",dataset="ISRUC",decompose_mode=None):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    
    # test_X = torch.permute(test_X,(0,2,1))
    
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)


    if decompose_mode == "seasonal":
        if mode!="train":
            train_sea_ = torch.load(data_path + f"train_{mode}_sea.pt")
        else:
            train_sea_ = torch.load(data_path + "train_sea.pt")
        test_sea_ = torch.load(data_path + "test_sea.pt")
        train_X_sea = train_sea_['samples']
        test_X_sea = test_sea_['samples']
    elif decompose_mode == "trend":
        if mode!="train":
            train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        else:
            train_trend_ = torch.load(data_path + "train_trend.pt")
        test_trend_ = torch.load(data_path + "test_trend.pt")
        train_X_sea = train_trend_['samples']
        test_X_sea = test_trend_['samples']
    elif decompose_mode == "resid":
        if mode!="train":
            train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        else:
            train_resid_ = torch.load(data_path + "train_resid.pt")
        test_resid_ = torch.load(data_path + "test_resid.pt")
        train_X_sea = train_resid_['samples']
        test_X_sea = test_resid_['samples']
    else:
        # 季节项
        print("to compute seasonal of train_X...")
        print(train_X.shape)
        train_X_sea = torch.permute(train_X,(0,2,1))
        sean_list = []
        for sample in train_X_sea :
            dims_list=[]
            for dims in sample:
                result = seasonal_decompose(dims, model='additive', period=30)
                trend = pd.Series(result.seasonal)
                trend = trend.ffill().bfill()
                trend = trend.to_numpy()
                dims_list.append(trend)
            dims_list = np.stack(dims_list)
            sean_list.append(dims_list)

        sean_list = np.stack(sean_list)
        train_X_sea = torch.Tensor(sean_list)
        print(train_X_sea.shape)
        train_X_sea = torch.permute(train_X_sea,(0,2,1))
        print(train_X_sea.shape)
        print("to compute seasonal of test_X...")
        print(test_X.shape)
        test_X_sea = torch.permute(test_X,(0,2,1))
        sean_list = []
        for sample in test_X_sea :
            dims_list=[]
            for dims in sample:
                result = seasonal_decompose(dims, model='additive', period=30)
                trend = pd.Series(result.seasonal)
                trend = trend.ffill().bfill()
                trend = trend.to_numpy()
                dims_list.append(trend)
            dims_list = np.stack(dims_list)
            sean_list.append(dims_list)

        sean_list = np.stack(sean_list)
        test_X_sea = torch.Tensor(sean_list)
        print(test_X_sea.shape)
        test_X_sea = torch.permute(test_X_sea,(0,2,1))
        print(test_X_sea.shape)


        scaler = StandardScaler()
        scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
        train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
        test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
        print(train_X.shape)
        print(train_X_fft.shape)
        print(train_X_sea.shape)
    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X_fft.shape)
    print("train_X_sea.shape",train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y


def load_tri_view(mode="train",dataset="ISRUC",decompose_mode=None):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    print(mode)
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
        
    print(data_path)
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']
    train_X = torch.transpose(train_X,2,1)
    print(train_X.shape)

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']
    print(decompose_mode)
    if decompose_mode == "seasonal":
        if mode!="train":
            train_sea_ = torch.load(data_path + f"train_{mode}_sea.pt")
        else:
            train_sea_ = torch.load(data_path + "train_sea.pt")
        test_sea_ = torch.load(data_path + "test_sea.pt")
        train_X_sea = train_sea_['samples']
        test_X_sea = test_sea_['samples']
        
        # if mode!="train":
        #     train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        # else:
        #     train_trend_ = torch.load(data_path + "train_trend.pt")
        # test_trend_ = torch.load(data_path + "test_trend.pt")
        # train_X_trend = train_trend_['samples']
        # test_X_trend = test_trend_['samples']

        # if mode!="train":
        #     train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        # else:
        #     train_resid_ = torch.load(data_path + "train_resid.pt")
        # test_resid_ = torch.load(data_path + "test_resid.pt")
        # train_X_resid = train_resid_['samples']
        # test_X_resid = test_resid_['samples']

    elif decompose_mode == "trend":
        if mode!="train":
            train_trend_ = torch.load(data_path + f"train_{mode}_trend.pt")
        else:
            train_trend_ = torch.load(data_path + "train_trend.pt")
        test_trend_ = torch.load(data_path + "test_trend.pt")
        train_X_sea = train_trend_['samples']
        test_X_sea = test_trend_['samples']
    elif decompose_mode == "resid":
        if mode!="train":
            train_resid_ = torch.load(data_path + f"train_{mode}_resid.pt")
        else:
            train_resid_ = torch.load(data_path + "train_resid.pt")
        test_resid_ = torch.load(data_path + "test_resid.pt")
        train_X_sea = train_resid_['samples']
        test_X_sea = test_resid_['samples']
    
    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
        # 频域
    print(type(train_X))
    print(train_X.shape)
    # train_X = torch.from_numpy(train_X)
    # test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    print("train_X.shape",train_X.shape)
    print("train_X_fft.shape",train_X.shape)
    print("train_X_sea.shape",train_X_sea.shape)
    #return [train_X,train_X_sea, train_X_fft], train_y, [test_X,test_X_sea,test_X_fft], test_y
    return [train_X_fft,train_X_sea, train_X], train_y, [test_X_fft,test_X_sea,test_X], test_y



def load_roadbank_two_view(mode="train",dataset="RoadBank"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    data_path = f"/workspace/CA-TCC/data/{dataset}/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    train_X = torch.transpose(train_X,2,1)
    # test_X = torch.permute(test_X,(0,2,1))
    print(train_X.shape)
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    # 季节项
    print("to compute seasonal of train_X...")
    print(train_X.shape)
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    print(train_X_sea.shape)
    print("to compute seasonal of test_X...")
    print(test_X.shape)
    test_X_sea = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X_sea = torch.Tensor(sean_list)
    print(test_X_sea.shape)
    test_X_sea = torch.permute(test_X_sea,(0,2,1))
    print(test_X_sea.shape)


    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
    test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
    print(train_X.shape)
    print(train_X_fft.shape)
    print(train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y



def load_uea_two_view(mode="train",dataset="SelfRegulationSCP1"):
    # train_X, train_y, test_X, test_y = load_HAR()
    # train_X_fft, _, test_X_fft, _ = load_HAR_fft()
    # train_X_sea, _, test_X_sea, _ = load_HAR_seasonal()
    dataset = "MotorImagery"
    data_path = f"/workspace/CA-TCC/data/UEA/{dataset}/"
    if mode!="train":
        train_ = torch.load(data_path + f"train_{mode}.pt")
    else:
        train_ = torch.load(data_path + "train.pt")
    # val_ = torch.load(data_path + "val.pt")
    
    train_X = train_['samples']
    # train = torch.transpose(train, 1, 2)
    train_y = train_['labels']

    test_ = torch.load(data_path + "test.pt")
    test = test_['samples']
    test_X = torch.transpose(test, 1, 2)
    test_y = test_['labels']

    # test_X = test
    # test_y = test_labels


    train_X = torch.transpose(train_X,2,1)
    # test_X = torch.permute(test_X,(0,2,1))
    print(train_X.shape)
    print(test_X.shape)
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    # 频域
    print(type(train_X))
    print(train_X.shape)
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)
    train_X_fft = torch.fft.fft(train_X.transpose(1, 2)).abs()
    test_X_fft = torch.fft.fft(test_X.transpose(1, 2)).abs()
    train_X_fft = train_X_fft.transpose(1, 2)
    test_X_fft = test_X_fft.transpose(1, 2)

    scaler = StandardScaler()
    scaler.fit(train_X_fft.reshape(-1, train_X_fft.shape[-1]))
    train_X_fft = scaler.transform(train_X_fft.reshape(-1, train_X_fft.shape[-1])).reshape(train_X_fft.shape)
    test_X_fft = scaler.transform(test_X_fft.reshape(-1, test_X_fft.shape[-1])).reshape(test_X_fft.shape)

    # 季节项
    print("to compute seasonal of train_X...")
    print(train_X.shape)
    train_X_sea = torch.permute(train_X,(0,2,1))
    sean_list = []
    for sample in train_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    train_X_sea = torch.Tensor(sean_list)
    print(train_X_sea.shape)
    train_X_sea = torch.permute(train_X_sea,(0,2,1))
    print(train_X_sea.shape)
    print("to compute seasonal of test_X...")
    print(test_X.shape)
    test_X_sea = torch.permute(test_X,(0,2,1))
    sean_list = []
    for sample in test_X_sea :
        dims_list=[]
        for dims in sample:
            result = seasonal_decompose(dims, model='additive', period=30)
            dims_list.append(result.seasonal)
        dims_list = np.stack(dims_list)
        sean_list.append(dims_list)

    sean_list = np.stack(sean_list)
    test_X_sea = torch.Tensor(sean_list)
    print(test_X_sea.shape)
    test_X_sea = torch.permute(test_X_sea,(0,2,1))
    print(test_X_sea.shape)


    scaler = StandardScaler()
    scaler.fit(train_X_sea.reshape(-1, train_X_sea.shape[-1]))
    train_X_sea = scaler.transform(train_X_sea.reshape(-1, train_X_sea.shape[-1])).reshape(train_X_sea.shape)
    test_X_sea = scaler.transform(test_X_sea.reshape(-1, test_X_sea.shape[-1])).reshape(test_X_sea.shape)
    print(train_X.shape)
    print(train_X_fft.shape)
    print(train_X_sea.shape)
    return [train_X, train_X_fft,train_X_sea], train_y, [test_X, test_X_fft,test_X_sea], test_y



def get_data_loader(args):
    if args.dataset in ["HAR","Epilepsy","ISRUC"]:
        train_data, train_labels, test_data, test_labels = load_isruc_two_view(args.data_perc,args.dataset,args.decomp_mode)

    if args.dataset == 'SleepEDF':
        train_data, train_labels, test_data, test_labels = load_EEG_two_view(args.data_perc)

    if args.dataset == 'RoadBank' or args.dataset == "Bridge":
        train_data, train_labels, test_data, test_labels = load_roadbank_two_view(args.data_perc,args.dataset)
    
    if args.dataset == 'Waveform':
        train_data, train_labels, test_data, test_labels = load_Waveform_two_view()
    if args.dataset == 'UEA':
        train_data, train_labels, test_data, test_labels = load_uea_two_view()
    
    return train_data, train_labels, test_data, test_labels