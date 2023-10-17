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

from nethope.models_multi import model
from nethope.model_multi_utils import test_performance, merge_result
from nethope.data_utils import get_pid_list, get_data, get_mlb, output_res, get_ppi_idx, get_homo_ppi_idx
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

    model_cnf['model']['model_path'] = Path(data_cnf['model_path'])/F'{run_name}'
    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])

    logger.info(F'Model: {model_name}, Path: {model_cnf["model"]["model_path"]}, Dataset: {data_name}')

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

    '''
    train_pid_list: protien list
        [protein1, protein2, protein3, ...]
    train_go: protein-go_list (list-list) 
        [[protein1's GO1, protein1's GO2, ...], 
         [protein2's GO1, protein2's GO2, ...], 
         ...]
    '''
    train_pid_list, train_go, train_esm = get_data(fasta_file = data_cnf['train']['fasta_file'],
                                                   pid_go_file = data_cnf['train']['pid_go_file'],
                                                   pid_esm_file = data_cnf['train']['esm_feature'])
    valid_pid_list, valid_go, valid_esm = get_data(fasta_file = data_cnf['valid']['fasta_file'],
                                                   pid_go_file = data_cnf['valid']['pid_go_file'],
                                                   pid_esm_file = data_cnf['valid']['esm_feature'])
    test_pid_list, test_go, test_esm = get_data(fasta_file = data_cnf['test']['fasta_file'],
                                                pid_go_file = data_cnf['test']['pid_go_file'],
                                                pid_esm_file = data_cnf['test']['esm_feature'])


    mlb = get_mlb(Path(data_cnf['mlb']), train_go)
    labels_num = len(mlb.classes_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y = mlb.transform(train_go).astype(np.float32)
        valid_y = mlb.transform(valid_go).astype(np.float32)
        test_y  = mlb.transform(test_go).astype(np.float32)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx

    logger.info(F'Number of Labels: {labels_num}')
    logger.info(F'Type of y: {type(train_y), type(valid_y)}')
    logger.info(F'Size of train y: {train_y.shape}')
    logger.info(F'Size of valid y: {valid_y.shape}')
    logger.info(F'Size of test y: {test_y.shape}')

    '''
    train_ppi: list, 将train_pid_list中的蛋白质转化为了net_pid_map中的编号，相当于dgl图中的节点编号
                     若net_pid_map中没有，则会被舍弃
    train_y: train_ppi对应的label
    '''
    *_, train_ppi, train_y, train_esm = get_ppi_idx(train_pid_list, train_y, net_pid_map, train_esm)
    '''
    valid_ppi: list, 将valid_pid_list中的蛋白质转化为了net_pid_map中的编号，相当于dgl图中的节点编号
                     若net_pid_map中没有，则用相似度最高的蛋白代替
    train_y: valid_ppi对应的label
    '''
    valid_res_idx_, valid_pid_list_, valid_ppi, valid_y, valid_esm = get_homo_ppi_idx(valid_pid_list, data_cnf['valid']['fasta_file'],
                                                                           valid_y, net_pid_map, valid_esm, net_blastdb,
                                                                           data_cnf['results']/F'{data_name}-valid-ppi-blast-out')
    '''
    test_res_idx_: [0, 1, 2, ...]
    test_pid_list_: [protein1, protein2, ...]
    test_ppi: [protein1在ppi中的编号, protein2在ppi中的编号, ...]
    test_y: go
    '''
    test_res_idx_, test_pid_list_, test_ppi, test_y, test_esm = get_homo_ppi_idx(test_pid_list, data_cnf['test']['fasta_file'],
                                                                       test_y, net_pid_map, test_esm, net_blastdb,
                                                                       data_cnf['results']/F'{data_name}-test-ppi-blast-out')
    scores = np.zeros((len(test_pid_list), len(mlb.classes_)))

    logger.info(F'Size of Training Set: {len(train_ppi)}')
    logger.info(F'Size of Validation Set: {len(valid_ppi)}')
    logger.info(F'Size of Test Set: {len(test_ppi)}')
    
    
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
    
    used_model_performance = np.array([0.0]*model_cnf['train']['ensemble_num'])
    
    for e in range(model_cnf['train']['epochs_num']):
        dataloader = dgl.dataloading.DataLoader(
            dgl_graph, train_ppi, sampler,
            batch_size = model_cnf['train']['batch_size'],
            shuffle = True,
            drop_last = False)

        train_loss_vals = AverageMeter()
        ppi_train_idx = np.full(network_x.shape[0], -1)
        ppi_train_idx[train_ppi] = np.arange(train_ppi.shape[0])

        test_model.train()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for input_nodes, output_nodes, blocks in tqdm(dataloader, leave=False, desc='Training Epoch {}: '.format(e)):
                blocks = [b.to(device) for b in blocks]
                
                input_features = (torch.from_numpy(network_x[input_nodes].indices).to(device).long(), 
                                  torch.from_numpy(network_x[input_nodes].indptr).to(device).long(), 
                                  torch.from_numpy(network_x[input_nodes].data).to(device).float())

                input_esms=[]
                for output_node in output_nodes:
                    input_esm = train_esm[ppi_train_idx[output_node]].numpy()
                    input_esms.append(input_esm)
                input_esms = np.array(input_esms)
                input_esms = input_esms.astype(np.float32)
                input_esms = torch.tensor(input_esms).to(device)

                output_labels = torch.from_numpy(train_y[ppi_train_idx[output_nodes]].toarray()).to(device)

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
                    
                    output_predictions = test_model(egg_blocks, egg_input_features, blocks, input_features, input_esms)

                    loss = loss_fn(output_predictions, output_labels)

                    train_loss_vals.update(loss.item(), len(output_nodes))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        valid_dataloader = dgl.dataloading.DataLoader(
            dgl_graph, valid_ppi, sampler,
            batch_size = model_cnf['test']['batch_size'],
            shuffle = True,
            drop_last = False)
        graph_fmax, graph_t, graph_aupr, plus_fmax, plus_Smin, plus_aupr, plus_t, df = test_performance(test_model, 
                                                                                                        valid_dataloader, 
                                                                                                        sampler,
                                                                                                        egg_graph,
                                                                                                        network_x,
                                                                                                        valid_ppi, 
                                                                                                        valid_pid_list_, 
                                                                                                        valid_y, 
                                                                                                        valid_esm, 
                                                                                                        idx_goid, goid_idx, 
                                                                                                        ont, device)
    
        print('Epoch: {}, Loss: {:.6f}, graph_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}'.format(e, 
                                                                                                                    train_loss_vals.avg, 
                                                                                                                    graph_fmax, 
                                                                                                                    graph_aupr, 
                                                                                                                    graph_t))
        print('Epoch: {}, Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, smin: {:.4f}, df_shape: {}'.format(e, 
                                                                                                                                 train_loss_vals.avg, 
                                                                                                                                 plus_fmax, 
                                                                                                                                 plus_aupr, 
                                                                                                                                 plus_t, plus_Smin, df.shape))

        if plus_aupr > min(used_model_performance):
            replace_ind = np.where(used_model_performance==min(used_model_performance))[0][0]
            used_model_performance[replace_ind] = plus_aupr
            torch.save({'epoch': e,'model_state_dict': test_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, 
                       './save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, replace_ind, model_cnf['train']['ensemble_num']))
        
        test_dataloader = dgl.dataloading.DataLoader(
            dgl_graph, test_ppi, sampler,
            batch_size = model_cnf['test']['batch_size'],
            shuffle = False,
            drop_last = False)
        graph_fmax, graph_t, graph_aupr, plus_fmax, plus_Smin, plus_aupr, plus_t, df  = test_performance(test_model, 
                                                                                                         test_dataloader, 
                                                                                                         sampler,
                                                                                                         egg_graph,
                                                                                                         network_x, 
                                                                                                         test_ppi, test_pid_list_, test_y,
                                                                                                         test_esm, 
                                                                                                         idx_goid, goid_idx, ont, device)
        print('\t\tgraph_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}'.format(graph_fmax, graph_aupr, graph_t))
        print('\t\tplus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, smin: {:.4f}, df_shape: {}'.format(plus_fmax, plus_aupr, 
                                                                                                                          plus_t, plus_Smin, df.shape))
    
    test_dataloader = dgl.dataloading.DataLoader(
        dgl_graph, test_ppi, sampler,
        batch_size = model_cnf['test']['batch_size'],
        shuffle = False,
        drop_last = False)
    cob_pred_df = []
    for i_t_min in range(model_cnf['train']['ensemble_num']):
        if os.path.exists('./save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, i_t_min, model_cnf['train']['ensemble_num'])):
            checkpoint = torch.load('./save_models/trans_attention_multi_best_{0}_{1}of{2}model.pt'.format(ont, i_t_min, model_cnf['train']['ensemble_num']))
            test_model.load_state_dict(checkpoint['model_state_dict'])
            pred_df = test_performance(test_model, test_dataloader, sampler, egg_graph, network_x, 
                                       test_ppi, test_pid_list_, test_y, test_esm, 
                                       idx_goid, goid_idx, ont, device, 
                                       save=True, 
                                       save_file='./results/trans_attention_multi_best_{0}_{1}of{2}model.pkl'.format(ont, i_t_min, model_cnf['train']['ensemble_num']), 
                                       evaluate=False)
            cob_pred_df.append(pred_df)
            print(i_t_min, pred_df.shape)
    final_result = merge_result(cob_pred_df)
    with open('./results/trans_attention_multi_best_{}_final.pkl'.format(ont), 'wb') as fw:
        pkl.dump(final_result, fw)
    print("Done")

if __name__ == '__main__':
    main()