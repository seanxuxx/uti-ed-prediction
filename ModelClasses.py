import numpy as np
import torch
import torch.nn as nn


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert len(X) == len(y), 'inconsistent shape between X and y'
        self.features = X
        self.labels = y
        self.length = len(X)
        self.n_feature = X.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        feature = torch.FloatTensor(self.features[i])
        label = torch.FloatTensor([self.labels[i]])
        return feature, label


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, X: np.ndarray):
        self.features = X
        self.length = len(X)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        feature = torch.FloatTensor(self.features[i])
        return feature


class NN(nn.Module):

    def __init__(self, input_size: int, dropout_rate: float):

        super(NN, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
