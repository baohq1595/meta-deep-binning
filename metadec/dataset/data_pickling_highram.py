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

def partial_build_graph_step1(reads, qmer_length, save_dir, n_procs=1):
    '''
    Build overlapping graph
    '''
    for i, r in enumerate(reads):
        reads[i] = reads[i].replace('N', '')

    # Create hash table with q-mers are keys
    # lmers_dict = build_hashtable(reads, qmer_length, save_dir, n_procs)
    lmers_dict=dict()
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            lmer = r[j:j+qmer_length]
            encoded_lmer = hash2numeric(lmer)
            if encoded_lmer in lmers_dict:
                lmers_dict[encoded_lmer] += [idx]
            else:
                lmers_dict[encoded_lmer] = [idx]

    print('Finish hashtable')

    # Building edges
    # E = build_edges(lmers_dict, save_dir, n_procs)
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

    return E

def partial_build_graph_step2(E, num_shared_reads, save_dir):
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    return E_Filtered


def partial_build_graph_step3(E_Filtered, labels, save_dir):
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to graph
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

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
                        maximum_seed_sizes, only_seed,
                        save_dir, pickle_dir, n_procs=1):
    # Read fasta dataset
    reads, labels = load_meta_reads(fna_file)

    # Build raw connection
    E = partial_build_graph_step1(reads, qmers, save_dir, n_procs)

    # Filter connection
    E_Filtered = partial_build_graph_step2(E, num_shared_reads, save_dir)
        
    # Build graph
    G = partial_build_graph_step3(E_Filtered, labels, save_dir)

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

    for maximum_seed_size in tqdm.tqdm(maximum_seed_sizes):
        groups, seeds = partition(G, maximum_seed_size)
        print(f'Finish partitioning. Size: {maximum_seed_size}')
        kmer_features = compute_feature(dictionary, corpus, groups, seeds, only_seed=only_seed)
        
        if is_normalize:
            scaler = StandardScaler()
            kmer_features = scaler.fit_transform(kmer_features)
        print('Finish compute feature')

        pickling(kmer_features, labels, groups, seeds, name, pickle_dir, maximum_seed_size)
        print('Pickling done')

    del reads, labels, dictionary, documents, corpus, groups, seeds, kmer_features

if __name__ == "__main__":
    # GROUP_SIZES = [1, 10, 25, 50, 75, 100, 150, 200, 500, 1000]
    GROUP_SIZES = [200, 500, 1000]
    DATASET_NAME = 'S1'
    PICKLE_DIR = f'data/{DATASET_NAME}/pickle'
    FASTA_FILE = f'data/simulated/raw/{DATASET_NAME}.fna'
    # FASTA_FILE = f'data/{DATASET_NAME}/raw/{DATASET_NAME}.fna'
    SAVE_DIR = f'data/{DATASET_NAME}/temp'
    KMERS = [4]
    LMER = 30
    # NUM_SHARED_READS = 5 if 'hmp' in DATASET_NAME else 45
    NUM_SHARED_READS = 5
    IS_NORMALIZE = True
    ONLY_SEED = True

    for d in [PICKLE_DIR, SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f'Processing {DATASET_NAME}...')
    print(f'Num shared reads {NUM_SHARED_READS}...')

    # for grsz in tqdm.tqdm(GROUP_SIZES):
    dataset_constructor(
        fna_file=FASTA_FILE,
        name=DATASET_NAME,
        kmers=KMERS,
        qmers=LMER, 
        num_shared_reads=NUM_SHARED_READS,
        is_normalize=IS_NORMALIZE,
        maximum_seed_sizes=GROUP_SIZES,
        only_seed=ONLY_SEED,
        save_dir=SAVE_DIR,
        pickle_dir=PICKLE_DIR,
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

