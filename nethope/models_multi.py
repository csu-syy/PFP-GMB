=import warnings
import click
import numpy as np
import scipy.sparse as ssp
import torch
import dgl
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger
from tqdm.auto import tqdm, trange
import networkx as nx
import torch.nn as nn
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F

class CustomGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, residual=True):
        super(CustomGraphConv, self).__init__()
        self.W = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, block, h):
        with block.local_scope():
            block.srcdata['h'] = h
            
            if self.residual:
                block.update_all(fn.u_mul_e('h', 'self', 'm_res'), 
                                 fn.sum('m_res', 'res'))
            
            block.update_all(fn.u_mul_e('h', 'ppi', 'ppi_m_out'), 
                             fn.sum('ppi_m_out', 'ppi_out'))
            
            h_dst = self.dropout(F.relu(self.W(block.dstdata['ppi_out'])))
            if self.residual:
                h_dst = h_dst + block.dstdata['res']
                
            return h_dst
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

        
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # in_size=75
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = F.softmax(w, dim=1)
        # print(z.size(), w.size(), beta.size())
        return (beta * z).sum(1), beta
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.project[0].weight)
        nn.init.xavier_uniform_(self.project[2].weight)
        

class model(nn.Module):
    def __init__(self, labels_num, input_size, embedding_size, hidden_size, num_gcn, dropout, residual):
        super(model, self).__init__()
        self.labels_num = labels_num
        
        self.embedding_layer = nn.EmbeddingBag(input_size, embedding_size, mode='sum', include_last_offset=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.ppi_gcn_layers = nn.ModuleList(CustomGraphConv(embedding_size, hidden_size, dropout=0.5, residual=residual) for _ in range(num_gcn))
        self.egg_gcn_layers = nn.ModuleList(CustomGraphConv(embedding_size, hidden_size, dropout=0.5, residual=residual) for _ in range(num_gcn))
        
        self.sequenceLayer=nn.Sequential(
            nn.Linear(1280, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(1024, hidden_size),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
        )
        
        self.attention = Attention(hidden_size, 256)
        
        self.trans_layer = nn.Linear(hidden_size*2, hidden_size*2)
        self.pred_layer = nn.Linear(hidden_size*2, labels_num)
        
        self.residual = residual
        self.num_gcn = num_gcn
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        for gcn in self.ppi_gcn_layers:
            gcn.reset_parameters()
        for gcn in self.egg_gcn_layers:
            gcn.reset_parameters()
        self.attention.reset_parameters()
        
        nn.init.xavier_uniform_(self.trans_layer.weight)
        nn.init.xavier_uniform_(self.pred_layer.weight)
    
    def forward(self, egg_blocks, egg_input_feature, blocks, input_feature, input_esm):
        
        # 处理序列特征
        allSequenceFeatures = self.sequenceLayer(input_esm)
        
        feature_embedding = self.dropout(F.relu(self.embedding_layer(*input_feature)))
        egg_feature_embedding = self.dropout(F.relu(self.embedding_layer(*egg_input_feature)))
        
        for i in range(self.num_gcn):
            feature_embedding = self.ppi_gcn_layers[i](blocks[i], feature_embedding)
            
        for i in range(self.num_gcn):
            egg_feature_embedding = self.egg_gcn_layers[i](egg_blocks[i], egg_feature_embedding)
        
        cob_features = torch.stack([feature_embedding, egg_feature_embedding], dim=1)
        cob_features, _ = self.attention(cob_features)
        
        all_features = torch.cat((allSequenceFeatures, cob_features), 1)
        
        all_features = F.relu(self.trans_layer(all_features))
        outputs = self.pred_layer(all_features)
        
        return outputs
