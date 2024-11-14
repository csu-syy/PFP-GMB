import warnings
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
import os
import pickle as pkl
import joblib

from nethope.models_multi import model
from nethope.model_multi_utils import test_performance_test, merge_result_test
from nethope.data_utils import get_pid_list, get_data_test, output_res, get_ppi_idx, get_homo_ppi_idx_test
from nethope.objective import AverageMeter
from nethope.evaluation import compute_performance

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']))
@click.option('-n', '--gpu-number', type=click.INT, default=0)

def main(data_cnf, gpu_number):
    yaml = YAML(typ='safe')
    ont = data_cnf
    data_cnf, model_cnf = yaml.load(Path('./configure/{}_multi.yaml'.format(data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
    device = torch.device('cuda:{}'.format(gpu_number))
    
    # data_name = bp mf cc 
    # model_name = DeepGraphGO
    data_name, model_name = data_cnf['name'], model_cnf['name'] 
    run_name = F'{model_name}-{data_name}'
    logger.info('run_name: {}'.format(run_name))

    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])

    net_pid_list = get_pid_list(data_cnf['network']['pid_list']) # [protein1, protein2, ...]
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)} # {protein1: 0, protein2: 1, ...}
    net_blastdb = data_cnf['network']['blastdb']

    dgl_graph = dgl.load_graphs(data_cnf['network']['dgl'])[0][0]
    egg_graph = dgl.load_graphs(data_cnf['network']['egg'])[0][0]

    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_:=np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    
    self_loop = torch.zeros_like(egg_graph.edata['ppi'])
    self_loop[egg_graph.edge_ids(nr_:=np.arange(egg_graph.number_of_nodes()), nr_)] = 1.0
    egg_graph.edata['self'] = self_loop

    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float()
    
    egg_graph.edata['ppi'] = egg_graph.edata['ppi'].float()
    egg_graph.edata['self'] = egg_graph.edata['self'].float()
    logger.info(F'{dgl_graph}, {dgl_graph.device}')

    network_x = ssp.load_npz(data_cnf['network']['feature'])
    logger.info(F'network_x type: {type(network_x)}\nnetwork_x size: {network_x.shape}')
    network_esm = np.load(data_cnf['network']['esm_ppi'])
    logger.info(F'network_esm type: {type(network_esm)}\nnetwork_esm size: {network_esm.shape}')

    test_pid_list, test_esm = get_data_test(fasta_file = data_cnf['test']['fasta_file'],
                                                pid_esm_file = data_cnf['test']['esm_feature'])


    mlb = joblib.load(Path(data_cnf['mlb']))
    labels_num = len(mlb.classes_)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx

    logger.info(F'Number of Labels: {labels_num}')

    '''
    test_res_idx_: [0, 1, 2, ...]
    test_pid_list_: [protein1, protein2, ...]
    test_ppi: [protein1在ppi中的编号, protein2在ppi中的编号, ...]
    '''
    test_res_idx_, test_pid_list_, test_ppi, test_esm = get_homo_ppi_idx_test(test_pid_list, data_cnf['test']['fasta_file'],
                                                                       net_pid_map, test_esm, net_blastdb,
                                                                       data_cnf['results']/F'{data_name}-test-ppi-blast-out')
    scores = np.zeros((len(test_pid_list), len(mlb.classes_)))
    
    test_model = model(labels_num = labels_num, 
                       input_size = network_x.shape[1], 
                       embedding_size = model_cnf['model']['hidden_size'], 
                       hidden_size = model_cnf['model']['hidden_size'], 
                       num_gcn = model_cnf['model']['num_gcn'], 
                       dropout = 0.5, 
                       residual=True)
    logger.info(test_model)

    test_model = test_model.to(device)
    optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model_cnf['model']['num_gcn'])
    
    test_dataloader = dgl.dataloading.DataLoader(
        dgl_graph, test_ppi, sampler,
        batch_size = model_cnf['test']['batch_size'],
        shuffle = False,
        drop_last = False)
    cob_pred_df = []
    for i_t_min in range(model_cnf['train']['ensemble_num']):
        logger.info('./save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, i_t_min, model_cnf['train']['ensemble_num']))
        if os.path.exists('./save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, i_t_min, model_cnf['train']['ensemble_num'])):
            checkpoint = torch.load('./save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, i_t_min, model_cnf['train']['ensemble_num']))
            test_model.load_state_dict(checkpoint['model_state_dict'])
            pred_df = test_performance_test(test_model, test_dataloader, sampler, egg_graph, network_x, 
                                       test_ppi, test_pid_list_, test_esm, 
                                       idx_goid, goid_idx, ont, device, 
                                       save=True, 
                                       save_file='./results/trans_attention_multi_best_{0}_{1}of{2}model.pkl'.format(ont, i_t_min, model_cnf['train']['ensemble_num']), 
                                       evaluate=False)
            cob_pred_df.append(pred_df)
            print(i_t_min, pred_df.shape)
            logger.info(i_t_min, pred_df.shape)
            logger.info("cob_pred_df", cob_pred_df)
    final_result = merge_result_test(cob_pred_df)
    with open('./results/test_{}_final.pkl'.format(ont), 'wb') as fw:
        pkl.dump(final_result, fw)
    print("Done")

if __name__ == '__main__':
    main()
