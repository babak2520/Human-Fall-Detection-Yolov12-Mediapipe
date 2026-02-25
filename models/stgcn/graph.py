"""
Graph utility for skeleton-based action recognition using MediaPipe pose landmarks.

Adapts the ST-GCN graph structure (originally for AlphaPose COCO format)
to work with MediaPipe's 33 pose landmarks by selecting 14 key joints.

Reference: https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
"""

import numpy as np


# MediaPipe landmark indices for the 14 key joints we use
# Maps our 14-node graph indices to MediaPipe landmark indices
MEDIAPIPE_TO_GRAPH = {
    0: 0,    # nose
    1: 12,   # right shoulder
    2: 11,   # left shoulder
    3: 14,   # right elbow
    4: 13,   # left elbow
    5: 16,   # right wrist
    6: 15,   # left wrist
    7: 24,   # right hip
    8: 23,   # left hip
    9: 26,   # right knee
    10: 25,  # left knee
    11: 28,  # right ankle
    12: 27,  # left ankle
    13: -1,  # mid-hip (computed as average of left and right hip)
}

# The 14 MediaPipe indices to extract (13 = computed mid-hip)
MEDIAPIPE_KEY_INDICES = [0, 12, 11, 14, 13, 16, 15, 24, 23, 26, 25, 28, 27]


class Graph:
    """Graph to model skeletons extracted by MediaPipe Pose.

    Adapted from the AlphaPose coco_cut layout to work with MediaPipe's
    33 pose landmarks, selecting 14 key joints that match the COCO skeleton
    structure used by ST-GCN.

    Args:
        strategy (str): Partitioning strategy for graph convolution.
            One of: 'uniform', 'distance', 'spatial'.
        layout (str): Skeleton layout. Currently supports 'mediapipe'.
        max_hop (int): Maximum distance between connected nodes.
        dilation (int): Spacing between kernel points.
    """

    def __init__(self, layout='mediapipe', strategy='spatial', max_hop=1, dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        self.get_adjacency(strategy)

    def get_edge(self, layout):
        """Define the skeleton graph edges.
        
        Our 14-node graph (matching coco_cut from the reference):
            0: nose
            1: right shoulder      2: left shoulder
            3: right elbow         4: left elbow
            5: right wrist         6: left wrist
            7: right hip           8: left hip
            9: right knee         10: left knee
           11: right ankle        12: left ankle
           13: mid-hip (center node)
        """
        if layout == 'mediapipe':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            # Same connectivity as coco_cut from the reference ST-GCN
            neighbor_link = [
                (6, 4), (4, 2), (2, 13),   # left arm -> left shoulder -> center
                (13, 1), (5, 3), (3, 1),    # center -> right shoulder, right arm
                (12, 10), (10, 8), (8, 2),  # left leg -> left hip -> left shoulder
                (11, 9), (9, 7), (7, 1),    # right leg -> right hip -> right shoulder
                (13, 0),                    # center -> nose
            ]
            self.edge = self_link + neighbor_link
            self.center = 13  # mid-hip as center node
        elif layout == 'coco_cut':
            # Original coco_cut for compatibility
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (6, 4), (4, 2), (2, 13),
                (13, 1), (5, 3), (3, 1),
                (12, 10), (10, 8), (8, 2),
                (11, 9), (9, 7), (7, 1),
                (13, 0),
            ]
            self.edge = self_link + neighbor_link
            self.center = 13
        else:
            raise ValueError(f'Layout "{layout}" is not supported!')

    def get_adjacency(self, strategy):
        """Compute adjacency matrix based on the chosen strategy."""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f'Strategy "{strategy}" is not supported!')


def get_hop_distance(num_node, edge, max_hop=1):
    """Compute the hop distance between all node pairs."""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    """Normalize a directed graph adjacency matrix."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    """Normalize an undirected graph adjacency matrix."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
