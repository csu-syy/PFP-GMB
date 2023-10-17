import warnings
import click
import numpy as np
import pandas as pd
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
import pickle as pkl

from nethope.evaluation import compute_performance, compute_performance_deepgoplus

def test_performance(test_model, test_dataloader, sampler, egg_graph, network_x, test_ppi_list, test_pid_list, targets, test_esm, idx_goid, goid_idx, ont, device, save=False, save_file=None, evaluate=True):
    test_model.eval()
    
    ppi_test_idx = np.full(network_x.shape[0], -1)
    ppi_test_idx[test_ppi_list] = np.arange(test_ppi_list.shape[0])
    
    true_labels = []
    pred_labels = []
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for input_nodes, output_nodes, blocks in tqdm(test_dataloader, leave=False, desc='Testing:'):
            blocks = [b.to(device) for b in blocks]

            input_features = (torch.from_numpy(network_x[input_nodes].indices).to(device).long(), 
                              torch.from_numpy(network_x[input_nodes].indptr).to(device).long(), 
                              torch.from_numpy(network_x[input_nodes].data).to(device).float())
            
            input_esms=[]
            for output_node in output_nodes:
                input_esm = test_esm[ppi_test_idx[output_node]].numpy()
                input_esms.append(input_esm)
            input_esms = np.array(input_esms)
            input_esms = input_esms.astype(np.float32)
            input_esms=torch.tensor(input_esms).to(device)

            output_labels = targets[ppi_test_idx[output_nodes]].toarray()
            
            egg_dataloader = dgl.dataloading.DataLoader(
                egg_graph, output_nodes, sampler,
                batch_size = len(output_nodes),
                shuffle = False,
                drop_last = False)
            for egg_input_nodes, egg_output_nodes, egg_blocks in egg_dataloader:
                egg_blocks = [b.to(device) for b in egg_blocks]
                egg_input_features = (torch.from_numpy(network_x[egg_input_nodes].indices).to(device).long(), 
                                      torch.from_numpy(network_x[egg_input_nodes].indptr).to(device).long(), 
                                      torch.from_numpy(network_x[egg_input_nodes].data).to(device).float())

            output_predictions = torch.sigmoid(test_model(egg_blocks, egg_input_features, blocks, input_features, input_esms)).detach().cpu().numpy()

            for predidx, pridx in enumerate(output_nodes):
                proteinid = test_pid_list[ppi_test_idx[pridx]]
                save_dict['protein_id'].append(proteinid)
                
                true_gos = set()
                for goidx in targets[ppi_test_idx[pridx]].indices:
                    true_gos.add(idx_goid[goidx])
                save_dict['gos'].append(true_gos)
                
                pred_gos = {}
                for goidx, goval in enumerate(output_predictions[predidx]):
                    pred_gos[idx_goid[goidx]] = goval
                save_dict['predictions'].append(pred_gos)

            true_labels.append(output_labels)
            pred_labels.append(output_predictions)
        
    true_labels = np.vstack(true_labels)
    pred_labels = np.vstack(pred_labels)
    
    df = pd.DataFrame(save_dict)
    if save:
        with open(save_file, 'wb') as fw:
            pkl.dump(df, fw)
    if evaluate:
        go_file = '/home/wbshi/work/yunyanshuai/benchmark/DataProcess/inputdata/go.obo'
        plus_fmax, plus_Smin, plus_aupr, plus_t, plus_recalls, plus_precisions = compute_performance_deepgoplus(df,go_file,ont)

        graph_fmax, graph_t, graph_aupr = compute_performance(true_labels, pred_labels, idx_goid, goid_idx, ont)

        return graph_fmax, graph_t, graph_aupr, plus_fmax, plus_Smin, plus_aupr, plus_t, df
    else:
        return df

def merge_result(cob_df_list):
    save_dict = {}
    save_dict['protein_id'] = []
    save_dict['gos'] = []
    save_dict['predictions'] = []
    
    for idx, row in cob_df_list[0].iterrows():
        save_dict['protein_id'].append(row['protein_id'])
        save_dict['gos'].append(row['gos'])
        pred_gos = {}
        # merge
        for go, score in row['predictions'].items():
            pred_gos[go] = score
        for single_df in cob_df_list[1:]:
            pred_scores = single_df[single_df['protein_id']==row['protein_id']].reset_index().loc[0, 'predictions']
            for go, score in pred_scores.items():
                pred_gos[go] += score
        # average
        avg_pred_gos = {}
        for go, score in pred_gos.items():
            avg_pred_gos[go] = score/len(cob_df_list)
        
        save_dict['predictions'].append(avg_pred_gos)
        
    df = pd.DataFrame(save_dict)
    
    return df