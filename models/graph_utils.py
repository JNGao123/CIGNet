from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, ii, jj = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]

    data = np.append(data, data)
    i = np.append(ii, jj)
    j = np.append(jj, ii)

    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)
   # a =  np.array(adj_mx.todense())
    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + 1.1*sp.eye(adj_mx.shape[0]))


    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


def adj_mx_from_skeleton2(cfg,num_joints = 32):

    num_joints = 2 * 32
    edges = getadj1(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def adj_mx_from_skeleton1(cfg,num_joints = 64):
    num_joints  = 2 * 32

    edges  = getadj(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def gatScaleAdj(cfg):
    return [adj_mx_from_skeleton16(cfg),adj_mx_from_skeleton16_nosys(cfg),adj_mx_from_skeleton8(cfg),adj_mx_from_skeleton8_nosys(cfg),adj_mx_from_skeleton4(cfg),adj_mx_from_skeleton4_nosys(cfg)]


################################16

#     return adj_mx_from_edges(num_joints, edges, sparse=False)
def adj_mx_from_skeleton16(cfg,num_joints = 64):
    num_joints  = 2 * 16

    edges  = getadj16(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def getadj16(cfg,numtemp = 16):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13), (13, 14), (14, 15),
             (0, 15), (1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin




def adj_mx_from_skeleton16_nosys(cfg,num_joints = 64):
    num_joints  = 2 * 16

    edges  = getadj16_nosys(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)



def getadj16_nosys(cfg,numtemp = 16):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13), (13, 14), (14, 15)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin


################################8

#     return adj_mx_from_edges(num_joints, edges, sparse=False)
def adj_mx_from_skeleton8(cfg,num_joints = 64):
    num_joints  = 2 * 8

    edges  = getadj8(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def getadj8(cfg,numtemp = 8):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),(0, 7), (1, 6), (2, 5), (3, 4)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin




def adj_mx_from_skeleton8_nosys(cfg,num_joints = 64):
    num_joints  = 2 * 8

    edges  = getadj8_nosys(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def getadj8_nosys(cfg,numtemp = 8):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin









####################################8

################################44444

#     return adj_mx_from_edges(num_joints, edges, sparse=False)
def adj_mx_from_skeleton4(cfg,num_joints = 64):
    num_joints  = 2 * 4

    edges  = getadj4(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def getadj4(cfg,numtemp = 4):
    edges = [(0, 1), (1, 2), (2, 3),(0, 3), (1, 2)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin




def adj_mx_from_skeleton4_nosys(cfg,num_joints = 64):
    num_joints  = 2 * 4

    edges  = getadj4_nosys(cfg)
    return adj_mx_from_edges(num_joints, edges, sparse=False)


def getadj4_nosys(cfg,numtemp = 4):
    edges = [(0, 1), (1, 2), (2, 3)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * numtemp for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * numtemp
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * numtemp)
                edgesFin.append(idx)

    return edgesFin









####################################8





def getadj(cfg):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13),
             (13, 14), (14, 15),
             (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25),
             (25, 26),
             (26, 27), (27, 28), (28, 29),(29,30),(30, 31),
             (0, 31), (1, 30), (2, 29), (3, 28), (4, 27), (5, 26), (6, 25), (7, 24), (8, 23), (9, 22), (10, 21),
             (11, 20),
             (12, 19), (13, 18), (14, 17), (15, 16)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * 32 for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * 32
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * 32)
                edgesFin.append(idx)

    return edgesFin




def getadj1(cfg):
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13),
             (13, 14), (14, 15),
             (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25),
             (25, 26),
             (26, 27), (27, 28), (28, 29),(29,30),(30, 31)]

    n = 2
    edgesFin = []

    for i in range(n):
        for j in range(len(edges)):
            idx = tuple(w + i * 32 for w in edges[j])

            edgesFin.append(idx)

    for i in range(n):
        start = i * 32
        end = n - i
        for j in range(start):
            for k in range(end):
                idx = (j, j + (k + 1) * 32)
                edgesFin.append(idx)

    return edgesFin