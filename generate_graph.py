import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import pickle as pkl

###此部分构建ppi_mat.npz
with open('./data/PPI_dict_protein_index.pkl','rb') as fr:
    dict_ppi_protein_index=pkl.load(fr)
    
# 获取所有的节点ID
rows = []
cols = []
data = []
dict_row_col={}
with open('./data/final_all_filter_300_graph_combine_score.txt','r') as f:
    for line in f.readlines():
        line=line.strip().split()
        if (line[0],line[1]) not in dict_row_col:
            dict_row_col[(line[0],line[1])]=1
        else:
            continue
        rows.append(int(dict_ppi_protein_index[line[0]]))
        cols.append(int(dict_ppi_protein_index[line[1]]))
        data.append(float(line[2])/1000)

num_nodes = len(dict_ppi_protein_index)
adj_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
save_npz('./data/single_ppi_mat.npz', adj_matrix)


rows = []
cols = []
data = []
dict_row_col={}
with open('./data/graph_eggnog.txt','r') as f:
    for line in f.readlines():
        line=line.strip().split()
        if (line[0],line[1]) not in dict_row_col:
            dict_row_col[(line[0],line[1])]=1
        else:
            if (line[1],line[0]) not in dict_row_col:
                dict_row_col[(line[1],line[0])]=1
                rows.append(int(dict_ppi_protein_index[line[1]]))
                cols.append(int(dict_ppi_protein_index[line[0]]))
                data.append(0.6)    
            continue
        rows.append(int(dict_ppi_protein_index[line[0]]))
        cols.append(int(dict_ppi_protein_index[line[1]]))
        data.append(0.6)
        if (line[1],line[0]) not in dict_row_col:
            dict_row_col[(line[1],line[0])]=1
            rows.append(int(dict_ppi_protein_index[line[1]]))
            cols.append(int(dict_ppi_protein_index[line[0]]))
            data.append(0.6) 

num_nodes = len(dict_ppi_protein_index)
adj_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
save_npz('./data/single_egg_mat.npz', adj_matrix)

'''
#此部分构建ppi_interpro.npz
with open('./data/PPI_dict_index_protein.pkl','rb') as fr:
    dict_ppi_index_protein=pkl.load(fr)

with open('./data/all_protein_interpro_new.pkl','rb') as fr:
    all_protein_interpro=pkl.load(fr)

rows = []
cols = []
data = []
for i in range(len(dict_ppi_index_protein)):
    protein=dict_ppi_index_protein[i]
    if protein not in all_protein_interpro:
        continue
    else:
        interpro=all_protein_interpro[protein]
    for item in interpro:
        rows.append(int(i))
        cols.append(int(item))
        data.append(1)
    
col_nodes = np.max(cols) + 1

ppi_interpro_matrix = csr_matrix((data, (rows, cols)), shape=(num_nodes,col_nodes))

### 保存为NPZ文件
save_npz('./data/ppi_interpro.npz', ppi_interpro_matrix)
'''