import click as ck
import numpy as np
import pandas as pd
import pickle as pkl
import pandas as pd
import numpy as np
import sys
import os
import gzip
import json
from collections import deque, Counter,defaultdict

AAS = {'M': 18, 'F': 15, 'A': 3, 'P': 9, 'I': 12, 'D': 11, 'V': 6, 'K': 7, 'S': 2, 'N': 13, 'W': 20, 'T': 10, 'G': 4, 'L': 1, 'Y': 16, 'E': 5, 'H': 17, 'R': 8, 'Q': 14, 'C': 19}

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

NAMESPACES_reverse = {
     'cellular_component':'cc',
     'molecular_function': 'mf',
     'biological_process':'bp',
}

evaluations = ['precision','recall','fmax','aupr']

class Ontology(object):

    def __init__(self, filename='./uniprot_sprot_train_test_data_oral/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        # print(annots[:10])
        # print(type(annots[0]))
        # sys.exit(0)
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        if it[0] == 'part_of':
                            obj['is_a'].append(it[1])
                            
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        if term_id in self.ont:
            return self.ont[term_id]['namespace']
        else:
            return 'can not find'
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

with open('/home/wbshi/work/yunyanshuai/benchmark/DataProcess/interpro_feature_PPI/PPI_dict_index_protein.pkl','rb') as fr:
    dict_ppi_index_protein=pkl.load(fr)
with open('./data/ppi_pid_list.txt','w') as fw:
    for i in range(len(dict_ppi_index_protein)):
        fw.write(dict_ppi_index_protein[i]+"\n")
    
f=open('../../DataProcess/inputdata/train_data_separate.pkl','rb')
train=pkl.load(f)

go_file='../../DataProcess/inputdata/go.obo'
go = Ontology(go_file, with_rels=True)

for class_type in ['test1','test2']:
    for class_tag in ['bp','cc','mf']:
        test_proteins_list=list(pd.read_csv('../../DataProcess/inputdata/{0}_data_separate_{1}_proteins.csv'.format(class_type,class_tag))['proteins']) 
        with open('../../DataProcess/inputdata/{0}_data_separate.pkl'.format(class_type),'rb') as fr:
            test_one=pkl.load(fr)
            
        with open('./data/{0}_{1}_pid_list.txt'.format(class_tag,class_type), 'w') as f: 
            for protein in test_proteins_list:
                f.write(f'{protein}\n')
        
        with open('./data/{0}_{1}_go.txt'.format(class_tag,class_type), 'w') as f:    
            for protein in test_proteins_list:
                value=test_one[protein]
                for go_id in value[f'all_{class_tag}']:
                    annot_set = go.get_anchestors(go_id)
                    for annot_id in annot_set:
                        if go.get_namespace(annot_id) == NAMESPACES[class_tag]:
                            f.write(f'{protein}\t{annot_id}\t{class_tag}\n')
                            
        with open('./data/{0}_{1}.fasta'.format(class_tag,class_type), 'w') as f:
            for protein in test_proteins_list:
                value=test_one[protein]
                seq=value['sequences']
                f.write(f'>{protein}\n{seq}\n')
                
for class_type in ['train']:
    for class_tag in ['bp','cc','mf']:
        train_all_proteins_list=list(pd.read_csv('../../DataProcess/inputdata/{0}_data_separate_{1}_proteins.csv'.format(class_type,class_tag))['proteins'])
        with open('./data/{0}_{1}_pid_list.txt'.format(class_tag,class_type), 'w') as f:
            for protein in train_all_proteins_list:
                f.write(f'{protein}\n')
        with open('./data/{0}_{1}_go.txt'.format(class_tag,class_type), 'w') as f:  
            for protein in train_all_proteins_list:
                value=train[protein]
                for go_id in value[f'all_{class_tag}']:
                    annot_set = go.get_anchestors(go_id)
                    for annot_id in annot_set:
                        if go.get_namespace(annot_id) == NAMESPACES[class_tag]:
                            f.write(f'{protein}\t{annot_id}\t{class_tag}\n')
        with open('./data/{0}_{1}.fasta'.format(class_tag,class_type), 'w') as f:
            for protein in train_all_proteins_list:
                value=train[protein]
                seq=value['sequences']
                f.write(f'>{protein}\n{seq}\n')