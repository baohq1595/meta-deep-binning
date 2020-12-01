import glob, os
import pickle
import tqdm

import networkx as nx
import nxmetis

from itertools import islice
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler

import sys
sys.path.append('.')
from metadec.dataset.utils import *
from metadec.utils.utils import *

def split_list(ori_list, chunks=2):
    batch_size = len(ori_list) // chunks
    splitted_list = []
    offset = 0
    for i in range(chunks):
        if i < chunks - 1:
            splitted_list.append(ori_list[offset:offset + batch_size])
            offset += batch_size
        else:
            splitted_list.append(ori_list[offset:])

    return splitted_list

def split_dict(ori_dict, chunks=2):
    batch_size = len(ori_dict) // chunks
    splitted_dict = []
    offset = 0
    for i in range(chunks):
        if i < chunks - 1:
            splitted_dict.append({k:ori_dict[k] for k in list(ori_dict.keys())[offset:offset + batch_size]})
            offset += batch_size
        else:
            splitted_dict.append({k:ori_dict[k] for k in list(ori_dict.keys())[offset:]})

    return splitted_dict

def chunkize(ori_dict, chunks=2):
    it = iter(ori_dict)
    batch_size = len(ori_dict) // chunks
    for i in range(0, len(ori_dict), batch_size):
        yield {k:ori_dict[k] for k in islice(it, batch_size)}

def build_overlap_graph_fast(reads, labels, qmer_length, num_shared_reads, n_procs=1):
    '''
    Build overlapping graph
    '''
    for i, r in enumerate(reads):
        reads[i] = reads[i].replace('N', '')
    # Create hash table with q-mers are keys
    lmers_dict = build_hashtable(reads, qmer_length, n_procs=n_procs)

    print('Finish hashtable')
    # Building edges
    E=dict()
    for encoded_lmer in lmers_dict:
        for e in it.combinations(lmers_dict[encoded_lmer],2):
            if e[0]!=e[1]:
                e_curr=(e[0],e[1])
            else:
                continue
            if e_curr in E:
                E[e_curr] += 1 # Number of connected lines between read a and b
            else:
                E[e_curr] = 1

def partial_build_graph_step1(reads, labels, qmer_length, num_shared_reads, save_dir, n_procs=1):
    '''
    Build overlapping graph
    '''
    for i, r in enumerate(reads):
        reads[i] = reads[i].replace('N', '')

    # Create hash table with q-mers are keys
    # lmers_dict = build_hashtable(reads, qmer_length, save_dir=save_dir, n_procs=n_procs)
    # lmers_dict = build_overlap_graph(reads, labels, qmer_length, num_shared_reads)
    lmers_dict=dict()
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            # lmer = r[j:j+qmer_length]
            lmer = hash2numeric(r[j:j+qmer_length])
            if lmer in lmers_dict:
                lmers_dict[lmer] += [idx]
            else:
                lmers_dict[lmer] = [idx]

    print('Finish hashtable')
    # Building edges
    E=dict()
    for encoded_lmer in lmers_dict:
        for e in it.combinations(lmers_dict[encoded_lmer],2):
            if e[0]!=e[1]:
                e_curr=(e[0],e[1])
            else:
                continue
            if e_curr in E:
                E[e_curr] += 1 # Number of connected lines between read a and b
            else:
                E[e_curr] = 1

    # Temporarily splitting into 4 parts
    with open(os.path.join(save_dir, 'edge_step1.pkl'), 'wb') as f:
        pickle.dump(E, f)

    print('Step 1. Finish!')
    
    return os.path.join(save_dir, 'edge_step1.pkl')

def partial_build_graph_step2(edge_file, labels, num_shared_reads, save_dir):
    with open(edge_file, 'rb') as f:
        E = pickle.load(f)

    # E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    # del E
    
    # with open(os.path.join(save_dir, 'edge_step2.pkl'), 'wb') as f:
    #     pickle.dump(E_Filtered, f)
        
    # return os.path.join(save_dir, 'edge_step2.pkl')
    #     with open(edge_file, 'rb') as f:
#         E = pickle.load(f)
    
#     sub_dicts = split_dict(E, 3)
    i = 0
    print('..........')
    for sub_chunk in chunkize(E, 10):
        with open(os.path.join(save_dir, f'E_{i}.pkl'), 'wb') as f:
            pickle.dump(sub_chunk, f)
        i+=1
    
        # with open(os.path.join(save_dir, 'E_2.pkl'), 'wb') as f:
        #     pickle.dump(sub_dicts[2], f)

        # with open(os.path.join(save_dir, 'E_0.pkl'), 'wb') as f:
        #     pickle.dump(sub_dicts[0], f)
        
    del E
    E_Filtered = {}
    for i in range(10):
        with open(os.path.join(save_dir, f'E_{i}.pkl'), 'rb') as f:
            E = pickle.load(f)
    
            temp = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
        E_Filtered.update(temp)
        del temp
    
    i = 0
    for sub_chunk in chunkize(E_Filtered, 10):
        with open(os.path.join(save_dir, f'E_Filtered_{i}.pkl'), 'wb') as f:
            pickle.dump(sub_chunk, f)
        i += 1
        
    del E_Filtered
        
    return os.path.join(save_dir, 'edge_step2.pkl')


def partial_build_graph_step3(edge_file, labels, num_shared_reads, save_dir):
    # with open(edge_file, 'rb') as f:
    #     E_Filtered = pickle.load(f)
        
    # sub_dicts = split_dict(E_Filtered, 3)
    # with open(os.path.join(save_dir, 'E_Filtered_1.pkl'), 'wb') as f:
    #     pickle.dump(sub_dicts[1], f)
    
    # with open(os.path.join(save_dir, 'E_Filtered_2.pkl'), 'wb') as f:
    #     pickle.dump(sub_dicts[2], f)
        
    # with open(os.path.join(save_dir, 'E_Filtered_0.pkl'), 'wb') as f:
    #     pickle.dump(sub_dicts[0], f)
    
    # del sub_dicts
    # del E_Filtered
    
    # # Initialize graph
    # G = nx.Graph()
    
    # # Add nodes to graph
    # for i in range(0, len(labels)):
    #     G.add_node(i, label=labels[i])
    
    
    # for i in range(3):
    #     print('Add edges')
    #     with open(os.path.join(save_dir, f'E_Filtered_{i}.pkl'), 'rb') as f:
    #         E_Filtered = pickle.load(f)

    #     # Add edges to graph
    #     for kv in E_Filtered.items():
    #         G.add_edge(kv[0][0], kv[0][1], weight=kv[1])
    #     del E_Filtered

    # # Finishing....
    # print('Finishing build graph')
    
    # return G
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to graph
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])
    
    
    for i in range(10):
        print('Add edges')
        with open(os.path.join(save_dir, f'E_Filtered_{i}.pkl'), 'rb') as f:
            E_Filtered = pickle.load(f)

        # Add edges to graph
        for kv in E_Filtered.items():
            G.add_edge(kv[0][0], kv[0][1], weight=kv[1])
        del E_Filtered

    # Finishing....
    print('Finishing build graph')
    
    return G

def partition(G, maximum_seed_size):
    return metis_partition_groups_seeds(G, maximum_seed_size)


def compute_feature(dictionary, corpus, groups, seeds, only_seed=True):
    return compute_kmer_dist(dictionary, corpus, groups, seeds, only_seed=only_seed)


def pickling(seed_kmer_features, labels, groups, seeds, name, save_path, group_size):
    save_dir = os.path.join(save_path, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = {
        'feature': seed_kmer_features,
        'label': labels,
        'group': groups,
        'seed': seeds,
        'name': name
    }

    with open(os.path.join(save_dir, f'data_{group_size}'), 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data['feature'],\
        data['label'],\
        data['group'],\
        data['seed']

def dataset_constructor(fna_file, name,
                        kmers: list, qmers, 
                        num_shared_reads, is_normalize,
                        maximum_seed_size, only_seed,
                        save_dir, n_procs=1):
    # Read fasta dataset
    reads, labels = load_meta_reads(fna_file)
    save_file_step1 = ''
    save_file_step2 = ''
    
    # Check and build edges for graph
    if not os.path.exists(os.path.join(save_dir, 'edge_step1.pkl')):
        save_file_step1 = partial_build_graph_step1(
                            reads, labels, qmers, 
                            num_shared_reads, save_dir, n_procs=n_procs
                        )
    
    # Load default file
    if save_file_step1 == '':
        save_file_step1 = os.path.join(save_dir, 'edge_step1.pkl')
    
    if not os.path.exists(os.path.join(save_dir, 'edge_step2.pkl')):
        save_file_step2 = partial_build_graph_step2(
            save_file_step1, labels, num_shared_reads, save_dir
        )
        
     # Load default file
    if save_file_step2 == '':
        save_file_step2 = os.path.join(save_dir, 'edge_step2.pkl')
        
    # Build graph
    G = partial_build_graph_step3(save_file_step2, labels, num_shared_reads, save_dir)
    groups, seeds = partition(G, maximum_seed_size)
    print('Finish partitioning')
    del G
    
    # If exists, load pickle
    if os.path.exists(os.path.join(save_dir, 'dict.pkl')):
        with open(os.path.join(save_dir, 'dict.pkl'), 'rb') as f:
            dictionary = pickle.load(f)
        with open(os.path.join(save_dir, 'documents.pkl'), 'rb') as f:
            documents = pickle.load(f)
        with open(os.path.join(save_dir, 'corpus.pkl'), 'rb') as f:
            corpus = pickle.load(f)
    else:        
        # Creating document from reads...
        dictionary, documents = create_document(reads, kmers)

        # Creating corpus...
        corpus = create_corpus(dictionary, documents, is_tfidf=False)
        # Save dict, corpus, docs
        with open(os.path.join(save_dir, 'dict.pkl'), 'wb') as f:
            pickle.dump(dictionary, f)

        with open(os.path.join(save_dir, 'documents.pkl'), 'wb') as f:
            pickle.dump(documents, f)

        with open(os.path.join(save_dir, 'corpus.pkl'), 'wb') as f:
            pickle.dump(corpus, f)
            
        
    
    kmer_features = compute_feature(dictionary, corpus, groups, seeds, only_seed=only_seed)
    
    if is_normalize:
        scaler = StandardScaler()
        kmer_features = scaler.fit_transform(kmer_features)
    print('Finish compute feature')

    pickling(kmer_features, labels, groups, seeds, name, save_dir, maximum_seed_size)
    print('Pickling done')

    del reads, labels, dictionary, documents, corpus, groups, seeds, kmer_features

if __name__ == "__main__":
    # GROUP_SIZES = [1, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
    GROUP_SIZES = [200, 500, 1000]
    DATASET_NAME = 'S1'
    PICKLE_DIR = f'data/simulated/{DATASET_NAME}/pickle'
    FASTA_FILE = f'data/simulated/raw/{DATASET_NAME}.fna'
    SAVE_DIR = f'data/simulated/temp'
    KMERS = [4]
    LMER = 30
    NUM_SHARED_READS = 5
    IS_NORMALIZE = True
    ONLY_SEED = True

    for d in [PICKLE_DIR, SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    for grsz in tqdm.tqdm_notebook(GROUP_SIZES):
        dataset_constructor(
            fna_file=FASTA_FILE,
            name=DATASET_NAME,
            kmers=KMERS,
            qmers=LMER, 
            num_shared_reads=NUM_SHARED_READS,
            is_normalize=IS_NORMALIZE,
            maximum_seed_size=grsz,
            only_seed=ONLY_SEED,
            save_dir=SAVE_DIR,
            n_procs=2
        )

        # seed_kmer_features, labels, groups, seeds = load_genomics(
        #         FASTA_FILE,
        #         kmers=KMERS,
        #         lmer=LMER,
        #         maximum_seed_size=grsz,
        #         num_shared_reads=NUM_SHARED_READS,
        #         is_deserialize=False,
        #         is_serialize=False,
        #         is_normalize=True,
        #         only_seed=ONLY_SEED,
        #         graph_file=os.path.join('temp', DATASET_NAME + '.json'),
        #         is_tfidf=False,
        #         is_amd=False,
        #         n_procs=4
        #     )

