import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
import pickle as pkl

#此部分构建ppi_interpro.npz
with open('./data/PPI_dict_index_protein.pkl','rb') as fr:
    dict_ppi_index_protein=pkl.load(fr)
    
with open('./data/esm_33_embeddings.pkl','rb') as fr:
    esm_33_embeddings=pkl.load(fr)

rows = []
cols = []
data = []
for i in range(len(dict_ppi_index_protein)):
    protein=dict_ppi_index_protein[i]
    if protein not in esm_33_embeddings:
        continue
    else:
        esm_feature=esm_33_embeddings[protein]
    for j in range(len(esm_feature)):
        rows.append(int(i))
        cols.append(int(j))
        data.append(float(esm_feature[j]))
    
col_nodes = np.max(cols) + 1
print(col_nodes)

ppi_interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(dict_ppi_index_protein),col_nodes))

### 保存为NPZ文件
save_npz('./data/ppi_esm_33.npz', ppi_interpro_matrix)