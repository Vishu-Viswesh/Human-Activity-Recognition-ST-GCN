import torch
import torch.nn as nn
from utils import Graph, normalize_adjacency

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency):
        super(SpatialGraphConv, self).__init__()
        self.register_buffer('A', torch.tensor(adjacency, dtype=torch.float32))
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x shape: (Batch, Channels, Temporal, Vertices)
        N, C, T, V = x.size()
        
        # Permute to map spatial operations smoothly: (N, T, V, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Linear projection across channels
        x = torch.matmul(x, self.weight) # (N, T, V, Out_C)
        
        # Spatial Graph Convolution via Adjacency Matrix
        # (N, T, V, Out_C) x (V, V) -> (N, T, V, Out_C)
        x = torch.einsum('ntvc,vw->ntwc', x, self.A)
        
        # Permute back to standard format: (Batch, Out_C, Temporal, Vertices)
        return x.permute(0, 3, 1, 2).contiguous()

class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, kernel_size=9, stride=1):
        super(STGCN_Block, self).__init__()
        self.sgcn = SpatialGraphConv(in_channels, out_channels, adjacency)
        
        # Temporal Convolution (1D convolution along the time dimension)
        padding = (kernel_size - 1) // 2
        self.tgcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3, inplace=True)
        )
        
        # Residual shortcut matching dimensions if needed
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.sgcn(x)
        x = self.tgcn(x)
        return self.relu(x + res)

class STGCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(STGCN, self).__init__()
        graph = Graph()
        A_norm = normalize_adjacency(graph.adjacency)
        
        # Feature extraction layers
        self.layer1 = STGCN_Block(in_channels, 64, A_norm)
        self.layer2 = STGCN_Block(64, 128, A_norm, stride=2)
        self.layer3 = STGCN_Block(128, 256, A_norm, stride=2)
        
        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Expected input shape: (Batch, Channels, Temporal_Frames, Vertices)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.global_pool(x) # Shape: (Batch, 256, 1, 1)
        x = torch.flatten(x, 1) # Shape: (Batch, 256)
        return self.fc(x)
