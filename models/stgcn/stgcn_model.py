"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for skeleton-based action recognition.

Adapted from: https://github.com/GajuuzZ/Human-Falling-Detect-Tracks
Original: https://github.com/yysijie/st-gcn

This implementation provides a two-stream architecture that processes both
joint positions (points) and motion vectors (frame-to-frame differences).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .graph import Graph


class GraphConvolution(nn.Module):
    """Graph convolution layer.

    Performs convolution over the spatial graph structure.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Number of adjacency matrix partitions (graph kernel size).
        t_kernel_size (int): Temporal kernel size for the convolution.
        t_stride (int): Temporal stride.
        t_padding (int): Temporal padding.
        t_dilation (int): Temporal dilation.
        bias (bool): Whether to use bias in the convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, V).
            A (torch.Tensor): Adjacency matrix of shape (K, V, V).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, T, V).
        """
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()


class st_gcn(nn.Module):
    """Spatial-Temporal Graph Convolution block.

    Applies graph convolution in space and temporal convolution in time,
    with residual connections, batch normalization, and ReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): (temporal_kernel_size, spatial_kernel_size).
        stride (int): Temporal stride.
        dropout (float): Dropout rate.
        residual (bool): Whether to use residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GraphConvolution(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class StreamSpatialTemporalGraph(nn.Module):
    """Single-stream ST-GCN for one input modality (points or motions).

    Architecture: 10 st_gcn layers (3→64→...→256), followed by global
    average pooling and a fully connected output layer.

    Args:
        in_channels (int): Number of input channels (3 for points, 2 for motions).
        graph_cfg (dict): Graph configuration (layout, strategy).
        num_class (int): Number of output classes.
        edge_importance_weighting (bool): Whether to learn edge importance.
        dropout (float): Dropout rate.
    """

    def __init__(self, in_channels=3, graph_cfg=None, num_class=7,
                 edge_importance_weighting=True, dropout=0.0):
        super().__init__()

        if graph_cfg is None:
            graph_cfg = {'layout': 'mediapipe', 'strategy': 'spatial'}

        graph = Graph(**graph_cfg)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input of shape (N, C, T, V).

        Returns:
            torch.Tensor: Class logits of shape (N, num_class).
        """
        N, C, T, V = x.size()

        x = x.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x


class TwoStreamSpatialTemporalGraph(nn.Module):
    """Two-stream ST-GCN for skeleton-based action recognition.

    Combines a points stream (joint positions) and a motions stream
    (frame-to-frame differences) for richer temporal modeling.

    Args:
        graph_cfg (dict): Graph configuration.
        num_class (int): Number of output classes.
        edge_importance_weighting (bool): Whether to learn edge importance.
        dropout (float): Dropout rate.
    """

    def __init__(self, graph_cfg=None, num_class=7,
                 edge_importance_weighting=True, dropout=0.0):
        super().__init__()

        if graph_cfg is None:
            graph_cfg = {'layout': 'mediapipe', 'strategy': 'spatial'}

        self.stream_points = StreamSpatialTemporalGraph(
            in_channels=3,
            graph_cfg=graph_cfg,
            num_class=num_class,
            edge_importance_weighting=edge_importance_weighting,
            dropout=dropout,
        )
        self.stream_motions = StreamSpatialTemporalGraph(
            in_channels=2,
            graph_cfg=graph_cfg,
            num_class=num_class,
            edge_importance_weighting=edge_importance_weighting,
            dropout=dropout,
        )

        self.fcn = nn.Linear(num_class * 2, num_class)

    def forward(self, points, motions):
        """Forward pass.

        Args:
            points (torch.Tensor): Joint positions of shape (N, 3, T, V).
            motions (torch.Tensor): Frame diffs of shape (N, 2, T, V).

        Returns:
            torch.Tensor: Class probabilities of shape (N, num_class).
        """
        out_pts = self.stream_points(points)
        out_mot = self.stream_motions(motions)

        out = torch.cat([out_pts, out_mot], dim=-1)
        out = self.fcn(out)
        out = torch.sigmoid(out)

        return out
