import time
from math import ceil
import os
import os.path as osp
import pickle
import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
from scipy import sparse as sp
import numpy as np
import networkx as nx
from data.map import unique_sign, unique_basis
from data.oap import oap_sign, oap_basis
from tqdm import tqdm

import torch
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
import argparse
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import sys
# Import EGNN model from the original code
from data.models.egnn import EGNNModel

class OGBMOLDGL(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = [g for g in data[self.split]]
        self.graph_lists = []
        self.graph_labels = []
        for g in self.data:
            if g[0].number_of_nodes() > 5:
                self.graph_lists.append(g[0])
                self.graph_labels.append(g[1])
        self.n_samples = len(self.graph_lists)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """
    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))
    return g


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    if EigVec.shape[0] <= pos_enc_dim:
        zeros = torch.zeros([EigVec.shape[0], pos_enc_dim - EigVec.shape[0] + 1]).float()
        g.ndata['pos_enc'] = torch.cat([g.ndata['pos_enc'], zeros], dim=-1)
    return g


def map_positional_encoding(g, pos_enc_dim, use_unique_sign=True, use_unique_basis=True, use_eig_val=True):
    """
        Graph positional encoding v/ Maximal Axis Projection
    """
    A = g.adjacency_matrix(scipy_fmt="csr").astype(np.double)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=np.double)
    A = torch.from_numpy((N * A * N).toarray())
    n, k = A.shape[0], pos_enc_dim
    E, U = torch.linalg.eigh(A)
    E, U = E.round(decimals=14)[:-1], U[:, :-1]
    dim = min(n - 1, k)
    _, mult = torch.unique(E[-dim:], return_counts=True)
    ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)]) + max(n - 1 - k, 0)
    if use_unique_sign:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                U[:, ind[i]:ind[i + 1]] = unique_sign(U[:, ind[i]:ind[i + 1]])  # eliminate sign ambiguity
    if use_unique_basis:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                continue  # single eigenvector, no basis ambiguity
            try:
                U[:, ind[i]:ind[i + 1]] = unique_basis(U[:, ind[i]:ind[i + 1]])  # eliminate basis ambiguity
            except AssertionError:
                continue  # assumption violated, skip
    if use_eig_val:
        Lambda = torch.nn.ReLU()(torch.diag(E))
        U = U @ torch.sqrt(Lambda)
    if n - 1 < k:
        zeros = torch.zeros([n, k - n + 1])
        U = torch.cat([U, zeros], dim=-1)
    g.ndata['pos_enc'] = U[:, -k:]  # last k non-trivial eigenvectors
    return g



def oap_positional_encoding(g, pos_enc_dim, use_unique_sign=True, use_unique_basis=True, use_eig_val=False):
    """
        Graph positional encoding v/ Orthogonalized Axis Projection
        # Laplacian
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N
    L = torch.from_numpy(L.toarray()).to(torch.float32)  # Convert to float32
    
    """
    A = g.adj_external(scipy_fmt="csr").astype(np.double)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=np.double)
    A = torch.from_numpy((N * A * N).toarray())
    n, k = A.shape[0], pos_enc_dim
    E, U = torch.linalg.eigh(A)
    E, U = E.round(decimals=14)[:-1], U[:, :-1]

        
    in_dim = 68
    dim = min(n - 1, k)
    _, mult = torch.unique(E[-dim:], return_counts=True)
    ind = torch.cat([torch.LongTensor([0]), torch.cumsum(mult, dim=0)]) + max(n - 1 - k, 0)
    
    # Find indices of eigenvectors with multiplicity 1
    single_ind = torch.tensor([ind[i] for i in range(mult.shape[0]) if mult[i] == 1])
    if single_ind.size(0) > 0:
      #create toch geometric data
      #change to fully connectrd graph
      edge_index = []
      for i in range(n):
          for j in range(n):
              if i != j:  # Don't include self-loops
                  edge_index.append([i, j])
      
      # Convert to tensor and reshape to [2, num_edges]
      edge_index = torch.tensor(edge_index, dtype=torch.long).t()
      
      #populate with single indices only
      wrapper = torch.zeros((n, pos_enc_dim), dtype=torch.double, device=U.device)
      
      wrapper[:, :single_ind.size(0)] = U[:, single_ind].clone()
      pos_single = wrapper
      eigvals_single = E[single_ind].clone()
      pyg_data = Data(pos=pos_single, eigvals=eigvals_single, edge_attr=g.edata['feat'], num_nodes=n)
      pyg_data.x = torch.zeros((n, in_dim), dtype=torch.long,device=pyg_data.pos.device)  # Dummy node features
      pyg_data.edge_index = to_undirected(edge_index)
      
      # Initialize EGNN model with the provided EGNNModel class
      egnn_base = EGNNModel(
          num_layers=2,
          emb_dim=in_dim,
          proj_dim=pos_enc_dim,
     ).to(torch.double)
      
      x_upd, h = egnn_base(pyg_data)
      x_upd = x_upd[:,:single_ind.size(0)].clone().detach()
  
      assert not torch.allclose(x_upd, pyg_data.pos.to(x_upd.device)[:,:single_ind.size(0)]), "EGNN output equals the eigenvectors input" 
      #check which eigenvectors are uncanonicalizable
      #U[:, single_ind] = x_upd
      # Calculate signs based on sum 
      sums = torch.sum(x_upd[:, :single_ind.size(0)], dim=0).round(decimals=6)  #TODO change back pos_single to x_upd
      if sums.dim() > 1:
          sums = sums.squeeze()
      
      sums = torch.sign(sums).to(x_upd.device)
      sums[sums == 0.] = torch.bernoulli(0.5*torch.ones(1).to(sums.device))  # Set the sign of zero to 1 TODO TEST THIS OCCURS NOT OFTEN
  
      #ones = torch.ones_like(sums)
      #if torch.allclose(sums, ones):
      #  print(f"Sums is too close to all ones vector.")
      #
      ## Create diagonal matrix of signs
      sums = torch.diag(sums)
      #
      ## Apply the sign correction (matrix multiplication as in original)
      U[:, single_ind] = pos_single[:, :single_ind.size(0)].detach() @ sums.to(pyg_data.pos.device).detach() #TODO uncomment
  

    if use_unique_sign:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                continue
                #U[:, ind[i]:ind[i + 1]] = oap_sign(U[:, ind[i]:ind[i + 1]].to(torch.double))  # eliminate sign ambiguity
    if use_unique_basis:
        for i in range(mult.shape[0]):
            if mult[i] == 1:
                continue  # single eigenvector, no basis ambiguity
            try:
                U[:, ind[i]:ind[i + 1]] = U[:, ind[i]:ind[i + 1]].to(torch.double)  # eliminate basis ambiguity
                continue
            except AssertionError:
                continue  # assumption violated, skip
    if use_eig_val:
        Lambda = torch.nn.ReLU()(torch.diag(E))
        U = U @ torch.sqrt(Lambda)
    if n - 1 < k:
        zeros = torch.zeros([n, k - n + 1])
        U = torch.cat([U, zeros], dim=-1)
    
    #wrapper = torch.zeros((pos_enc_dim), dtype=torch.double, device=U.device)
    #wrapper[-single_ind.size(0):] = eigvals_single
    g.ndata['pos_enc'] = pos_single[:, -k:]  # last k non-trivial eigenvectors
    #g.ndata['eig_val'] = eigvals_single[-k:].repeat(n, 1) #last k non-trivial eigenvalues
    #add hidden state faetures
    #temp_zeros = torch.zeros(len(g.ndata['feat']), 1 + in_dim)
    #temp_zeros[:, 0]  = g.ndata['feat']
    #temp_zeros[:, 1:] = h.detach()
    #g.ndata['feat']   = temp_zeros
    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    n = g.number_of_nodes()
    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adjacency_matrix(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float)  # D^-1
        RW = A * Dinv
        M = RW
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc - 1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE, dim=-1)
        g.ndata['pos_enc'] = PE
    return g


def make_full_graph(graph, adaptive_weighting=None):
    g, label = graph
    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    # Copy over the node feature data and laplace eigvecs
    full_g.ndata['feat'] = g.ndata['feat']
    try:
        full_g.ndata['pos_enc'] = g.ndata['pos_enc']
    except:
        pass
    try:
        full_g.ndata['eigvec'] = g.ndata['eigvec']
    except:
        pass
    # Initialize fake edge features w/ 0s
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges(), 3, dtype=torch.long)
    full_g.edata['real'] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    # Copy real edge data over, and identify real edges!
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['feat'] = g.edata['feat']
    full_g.edges[g.edges(form='uv')[0].tolist(), g.edges(form='uv')[1].tolist()].data['real'] = torch.ones(
        g.edata['feat'].shape[0], dtype=torch.long)  # This indicates real edges
    # This code section only apply for GraphiT --------------------------------------------
    if adaptive_weighting is not None:
        p_steps, gamma = adaptive_weighting
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        # Adaptive weighting k_ij for each edge
        if p_steps == "qtr_num_nodes":
            p_steps = int(0.25 * n)
        elif p_steps == "half_num_nodes":
            p_steps = int(0.5 * n)
        elif p_steps == "num_nodes":
            p_steps = int(n)
        elif p_steps == "twice_num_nodes":
            p_steps = int(2 * n)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        I = sp.eye(n)
        L = I - N * A * N
        k_RW = I - gamma * L
        k_RW_power = k_RW
        for _ in range(p_steps - 1):
            k_RW_power = k_RW_power.dot(k_RW)
        k_RW_power = torch.from_numpy(k_RW_power.toarray())
        # Assigning edge features k_RW_eij for adaptive weighting during attention
        full_edge_u, full_edge_v = full_g.edges()
        num_edges = full_g.number_of_edges()
        k_RW_e_ij = []
        for edge in range(num_edges):
            k_RW_e_ij.append(k_RW_power[full_edge_u[edge], full_edge_v[edge]])
        full_g.edata['k_RW'] = torch.stack(k_RW_e_ij, dim=-1).unsqueeze(-1).float()
    # --------------------------------------------------------------------------------------
    return full_g, label


class OGBMOLDataset(Dataset):
    def __init__(self, name, features='full'):
        start = time.time()
        self.name = name.lower()
        self.dataset = DglGraphPropPredDataset(name=self.name)
        if features == 'full':
            pass
        elif features == 'simple':
            # only retain the top two node/edge features
            for g in self.dataset.graphs:
                g.ndata['feat'] = g.ndata['feat'][:, :2]
                g.edata['feat'] = g.edata['feat'][:, :2]
        split_idx = self.dataset.get_idx_split()
        self.train = OGBMOLDGL(self.dataset, split_idx['train'])
        self.val = OGBMOLDGL(self.dataset, split_idx['valid'])
        self.test = OGBMOLDGL(self.dataset, split_idx['test'])
        self.evaluator = Evaluator(name=self.name)
        self.train_len = len(self.train)
        self.val_len = len(self.val)
        self.test_len = len(self.test)

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        labels = torch.stack(labels)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        return batched_graph, labels, snorm_n

    def _add_lap_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.train]
        self.val = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(lap_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]

    def _add_eig_vecs(self, pos_enc_dim):
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.train]
        self.val = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.val]
        self.test = [(add_eig_vec(g, pos_enc_dim), label) for g, label in self.test]

    def _init_positional_encodings(self, pos_enc_dim, type_init):
        # Initializing positional encoding randomly with l2-norm 1
        self.train = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.train]
        self.val = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.val]
        self.test = [(init_positional_encoding(g, pos_enc_dim, type_init), label) for g, label in self.test]

    def _add_use_positional_encodings(self, pos_enc_dim):
        # Graph positional encoding v/ Maximal Axis Projection
        if self.name in ['ogbg-molpcba', 'ogbg-molhiv', 'ogbg-moltoxcast', 'ogbg-mollipo']:  # too large
            cache_dir = f"cache/{self.name}/"
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            if osp.exists(f"cache/{self.name}/k={pos_enc_dim}.pkl"):
                with open(f"cache/{self.name}/k={pos_enc_dim}.pkl", "rb") as f:
                    train_cache, val_cache, test_cache = pickle.load(f)
                for i in range(self.train_len):
                    self.train[i][0].ndata['pos_enc'] = train_cache[i]
                for i in range(self.val_len):
                    self.val[i][0].ndata['pos_enc'] = val_cache[i]
                for i in range(self.test_len):
                    self.test[i][0].ndata['pos_enc'] = test_cache[i]
            else:
                self.train = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.train]
                self.val = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
                self.test = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]
                train_cache = [g.ndata['pos_enc'] for g, _ in self.train]
                val_cache = [g.ndata['pos_enc'] for g, _ in self.val]
                test_cache = [g.ndata['pos_enc'] for g, _ in self.test]
                cache = (train_cache, val_cache, test_cache)
                with open(f"cache/{self.name}/k={pos_enc_dim}.pkl", "wb") as f:
                    pickle.dump(cache, f)
        else:
            cache_path = f"cache/{self.name}/k={pos_enc_dim}.pkl"
            cache_dir = f"cache/{self.name}/"
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            if osp.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.train, self.val, self.test = pickle.load(f)
            else:
                self.train = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.train]
                self.val = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
                self.test = [(map_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]
                cache = (self.train, self.val, self.test)
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)

    def _add_oap_positional_encodings(self, pos_enc_dim):
        cache_dir = f"cache/{self.name}_oap/"
        if not osp.exists(cache_dir):
            os.makedirs(cache_dir)
        if osp.exists(f"cache/{self.name}_oap/k={pos_enc_dim}.pkl"):
            with open(f"cache/{self.name}_oap/k={pos_enc_dim}.pkl", "rb") as f:
                train_cache, val_cache, test_cache = pickle.load(f)
            for i in range(self.train_len):
                self.train[i][0].ndata['pos_enc'] = train_cache[i]
            for i in range(self.val_len):
                self.val[i][0].ndata['pos_enc'] = val_cache[i]
            for i in range(self.test_len):
                self.test[i][0].ndata['pos_enc'] = test_cache[i]
        else:
            self.train = [(oap_positional_encoding(g, pos_enc_dim), label) for g, label in self.train]
            self.val = [(oap_positional_encoding(g, pos_enc_dim), label) for g, label in self.val]
            self.test = [(oap_positional_encoding(g, pos_enc_dim), label) for g, label in self.test]
            train_cache = [g.ndata['pos_enc'] for g, _ in self.train]
            val_cache = [g.ndata['pos_enc'] for g, _ in self.val]
            test_cache = [g.ndata['pos_enc'] for g, _ in self.test]
            cache = (train_cache, val_cache, test_cache)
            with open(f"cache/{self.name}_oap/k={pos_enc_dim}.pkl", "wb") as f:
                pickle.dump(cache, f)

    def _make_full_graph(self, adaptive_weighting=None):
        self.train = [make_full_graph(graph, adaptive_weighting) for graph in self.train]
        self.val = [make_full_graph(graph, adaptive_weighting) for graph in self.val]
        self.test = [make_full_graph(graph, adaptive_weighting) for graph in self.test]
