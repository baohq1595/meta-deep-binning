from bz2 import compress
import itertools as it
import numpy as np
import networkx as nx
import nxmetis
import copy

from Bio import SeqIO
from Bio.Seq import Seq
import re
import gensim
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel

import pickle
import os
from stellargraph import StellarGraph
import pandas as pd
import stellargraph
from torch import long

def load_amd_reads(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    reads = []
    raw_labels = []
    labels = []
    read_frags = []
    for k, line in enumerate(lines):
        line = line.strip()
        if '>' in line or k == len(lines) - 1:
            if len(read_frags) > 0:
                reads.append(''.join(read_frags))
                read_frags = []
            if '>' in line:
                raw_label = line.split('|')[-1]
                raw_labels.append(raw_label)
        else:
            read_frags.append(line)

    unique_label = list(set(raw_labels))
    # label2idx = {}
    # cur_index = 0
    # for label in raw_labels:
    #     if label not in label2idx:
    #         label2idx[label] = cur_index
    #         cur_index += 1

    label2idx = {label: i for i, label in enumerate(unique_label)}
    labels = [label2idx[label] for label in raw_labels]
    return reads, labels, label2idx


def load_meta_reads(filename, type='fasta'):
    def format_read(read):
        # Return sequence and label
        z = re.split('[|={,]+', read.description)
        return read.seq, z[3]
    try:
        seqs = list(SeqIO.parse(filename, type))
        reads = []
        labels = []

        # Detect for paired-end or single-end reads
        # If the id of two first reads are different (e.g.: .1 and .2), they are paired-end reads
        is_paired_end = False
        if len(seqs) > 2 and seqs[0].id[-1:] != seqs[1].id[-1:]:
            is_paired_end = True

        label_list = dict()
        label_index = 0

        for i in range(0, len(seqs), 2 if is_paired_end else 1):
            read, label = format_read(seqs[i])
            if is_paired_end:
                read2, label2 = format_read(seqs[i + 1])
                read += read2
            reads += [str(read)]

            # Create labels
            if label not in label_list:
                label_list[label] = label_index
                label_index += 1
            labels.append(label_list[label])

        del seqs

        return reads, labels, label_list
    except Exception as e:
        print('Error when loading file {} '.format(filename))
        print('Cause: ', e)
        return None

def gen_kmers(klist):
    '''
    Generate list of k-mer words. Given multiple k-mer values.
    Args:
        klist: list of k-mer value
    Return:
        List of k-mer words
    '''
    bases = ['A', 'C', 'G', 'T']
    kmers_list = []
    for k in klist:
        kmers_list += [''.join(p) for p in it.product(bases, repeat=k)]

    # reduce a half of k-mers due to symmetry
    kmers_dict = dict()
    for myk in kmers_list:
        k_reverse_complement=Seq(myk).reverse_complement()
        if not myk in kmers_dict and not str(k_reverse_complement) in kmers_dict:
            kmers_dict[myk]=0

    return list(kmers_dict.keys())

def create_document( reads, klist):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param klist: list of int
    :return: list of str
    """
    # create a set of document
    documents = []
    for read in reads:
        k_mers_read = []
        for k in klist:
            k_mers_read += [read[j:j + k] for j in range(0, len(read) - k + 1)]
        documents.append(k_mers_read)

    k_mers_set = [gen_kmers(klist)]
    dictionary = corpora.Dictionary(k_mers_set)
    return dictionary, documents

def save_documents(documents, file_path):
    with open(file_path, 'w') as f:
        for d in documents:
            f.write("%s\n" % d)


def parallel_create_document(reads, klist, n_workers=2 ):
    """
    Create a set of document from reads, consist of all k-mer in each read
    For example:
    k = [3, 4, 5]
    documents =
    [
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 1
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read 2
        ...
        'AAA AAT ... AAAT AAAC ... AAAAT AAAAC' - read n
    ]
    :param reads:
    :param klist: list of int
    :return: list of str
    """

    # create k-mer dictionary
    k_mers_set = [gen_kmers( klist )] #[genkmers(val) for val in klist]
    dictionary = corpora.Dictionary(k_mers_set)

    documents = []
    reads_str_chunk = [list(item) for item in np.array_split(reads, n_workers)]
    chunks = [(reads_str_chunk[i], klist) for i in range(n_workers)]
    pool = Pool(processes=n_workers)

    result = pool.starmap(create_document, chunks)
    for item in result:
        documents += item
    return dictionary, documents

def create_corpus(dictionary: corpora.Dictionary, documents, 
                  is_tfidf=False, 
                  smartirs=None, 
                  is_log_entropy=False, 
                  is_normalize=True):
    corpus = [dictionary.doc2bow(d, allow_update=False) for d in documents]
    if is_tfidf:
        tfidf = TfidfModel(corpus=corpus, smartirs=smartirs)
        corpus = tfidf[corpus]
    elif is_log_entropy:
        log_entropy_model = LogEntropyModel(corpus, normalize=is_normalize)
        corpus = log_entropy_model[corpus]

    return corpus

def compute_kmer_dist(dictionary, corpus, groups, seeds, only_seed=True):
    # print('Start computing feature')
    corpus_m = gensim.matutils.corpus2dense(corpus, len(dictionary.keys())).T
    res = []
    if only_seed:
        for seednodes in seeds:
            tmp = corpus_m[seednodes, :]
            res += [np.mean(tmp, axis=0)]
    else:
        for groupnodes in groups:
            tmp = corpus_m[groupnodes, :]
            res += [np.mean(tmp, axis=0)]
    return np.array(res)

def build_hash_table(reads, qmer_length):
    lmers_dict=dict()
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            lmer = hash(r[j:j+qmer_length])
            if lmer in lmers_dict:
                lmers_dict[lmer] += [idx]
            else:
                lmers_dict[lmer] = [idx]
        
    return lmers_dict

def find_overlap_lmer(lmers_dict, num_shared_reads):
    E=dict()
    for lmer in lmers_dict:
        for e in it.combinations(lmers_dict[lmer],2):
            if e[0]!=e[1]:
                e_curr=(e[0],e[1])
            else:
                continue
            if e_curr in E:
                E[e_curr] += 1 # Number of connected lines between read a and b
            else:
                E[e_curr] = 1
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}

    return E_Filtered


def build_overlap_graph(reads, labels, qmer_length, num_shared_reads):
    '''
    Build overlapping graph
    '''
    # Create hash table with q-mers are keys
    print('Building hashtable...')
    lmers_dict = build_hash_table(reads, qmer_length)

    
    # Building edges
    print('Finding overlapped lmer...')
    E_Filtered = find_overlap_lmer(lmers_dict, num_shared_reads)
    
    print('Start initializing graph...')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes!!!')
    # Add nodes to graph
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    print('Add edges!!!')
    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

    # Finishing....
    print('Finishing build graph.')
    
    return G

def dump_hashtable(lmers_dict, parts, pickle_dir, comp):
    from compress_pickle import dump as dump_pickle
    batch_size = len(lmers_dict) // parts
    offset = 0
    pickle_paths = []
    for i in range(parts):
        if i < parts - 1:
            cur_dict = {k:lmers_dict[k] for k in list(lmers_dict.keys())[offset:offset + batch_size]}
            offset += batch_size
        else:
            cur_dict = {k:lmers_dict[k] for k in list(lmers_dict.keys())[offset:]}

        pickle_path = os.path.join(pickle_dir, f'hash_pkl_{i}.dat')
        pickle_paths.append(pickle_path)
        with open(pickle_path, 'wb') as f:
            dump_pickle(cur_dict, f, compression=comp, set_default_extension=False)
    
    return pickle_paths

def dump_edge_table(edge_dict, parts, pickle_dir, comp):
    from compress_pickle import dump as dump_pickle
    batch_size = len(edge_dict) // parts
    offset = 0
    pickle_paths = []
    for i in range(parts):
        if i < parts - 1:
            cur_dict = {k:edge_dict[k] for k in list(edge_dict.keys())[offset:offset + batch_size]}
            offset += batch_size
        else:
            cur_dict = {k:edge_dict[k] for k in list(edge_dict.keys())[offset:]}

        pickle_path = os.path.join(pickle_dir, f'edge_pkl_{i}.dat')
        pickle_paths.append(pickle_path)
        with open(pickle_path, 'wb') as f:
            dump_pickle(cur_dict, f, compression=comp, set_default_extension=False)
    
    return pickle_paths

def build_overlap_graph_low_mem(reads, labels, qmer_length, num_shared_reads, parts=100, comp='gzip'):
    '''
    Build overlapping graph
    '''
    from compress_pickle import dump, load
    import glob

    pickle_dir = 'temp/pickle/hashtable'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    pickle_paths = glob.glob(pickle_dir + '/*.dat')

    if len(pickle_paths) < parts:
        # Create hash table with q-mers are keys
        print('Building hashtable...')
        lmers_dict = build_hash_table(reads, qmer_length)

        # pickle and compress
        pickle_paths = dump_hashtable(lmers_dict, parts, pickle_dir, comp)
    
        lmers_dict = None
        
    # Building edges
    print('Finding overlapped lmer...')
    pickle_edge_dir = 'temp/pickle/edge'
    if not os.path.exists(pickle_edge_dir):
        os.makedirs(pickle_edge_dir)

    pickle_edge_paths = glob.glob(pickle_edge_dir + '/*.dat')

    E=dict()
    # If not found enough pickle files, then build and dump edge dict
    if len(pickle_edge_paths) < parts:
        # Load hash table pickle files to build edge dict
        for pickle_path in pickle_paths:
            lmers_dict = load(pickle_path, compression=comp, set_default_extension=False)
            for lmer in lmers_dict:
                for e in it.combinations(lmers_dict[lmer], 2):
                    if e[0]!=e[1]:
                        e_curr=(e[0],e[1])
                    else:
                        continue
                    if e_curr in E:
                        E[e_curr] += 1 # Number of connected lines between read a and b
                    else:
                        E[e_curr] = 1
        E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
        E = None
        # dump edge dict
        print('Serialize E_Filtered to pickle...')
        pickle_edge_paths = dump_edge_table(E_Filtered, parts, pickle_edge_dir, comp)
        E_Filtered = None

    print('Start initializing graph...')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes!!!')
    # Add nodes to graph
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    print('Add edges!!!')
    # Add edges to graph
    for i, pickle_edge_path in enumerate(pickle_edge_paths):
        E_Filtered = load(pickle_edge_path, compression=comp, set_default_extension=False)
        print(f'Loaded {i+1} dict with {len(E_Filtered)} edges.')

        for kv in E_Filtered.items():
            G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

    # Finishing....
    print('Finishing build graph.')
    
    return G

def create_dataframe_for_graph(edge_dict, labels):
    square_weighted_edges = pd.DataFrame(
    {
        "source": [e[0] for e in edge_dict.keys()],
        "target": [e[1] for e in edge_dict.keys()],
        "weight": [w for w in edge_dict.values()],
    })

    print(square_weighted_edges['source'].max())
    print(square_weighted_edges['target'].max())

    square_node_data = pd.DataFrame(
        {"label": [l for l in labels]},
        index=[i for i in range(len(labels))]
    )

    stellar_graph = StellarGraph(square_node_data, square_weighted_edges)

    return stellar_graph

def build_overlap_stellar_graph_low_mem(reads, labels, qmer_length, num_shared_reads, parts=100, comp='gzip'):
    '''
    Build overlapping graph
    '''
    from compress_pickle import dump, load
    import glob

    pickle_dir = 'temp/pickle/hashtable'
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)

    pickle_paths = glob.glob(pickle_dir + '/*.dat')

    if len(pickle_paths) < parts:
        # Create hash table with q-mers are keys
        print('Building hashtable...')
        lmers_dict = build_hash_table(reads, qmer_length)

        # pickle and compress
        pickle_paths = dump_hashtable(lmers_dict, parts, pickle_dir, comp)
    
        lmers_dict = None
        
    # Building edges
    print('Finding overlapped lmer...')
    E=dict()
    # Load hash table pickle files to build edge dict
    for pickle_path in pickle_paths:
        lmers_dict = load(pickle_path, compression=comp, set_default_extension=False)
        for lmer in lmers_dict:
            for e in it.combinations(lmers_dict[lmer], 2):
                if e[0]!=e[1]:
                    e_curr=(e[0],e[1])
                else:
                    continue
                if e_curr in E:
                    E[e_curr] += 1 # Number of connected lines between read a and b
                else:
                    E[e_curr] = 1
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    E = None

    print('Start initializing stellar graph...')
    G = create_dataframe_for_graph(E_Filtered, labels)
    
    # Finishing....
    print('Finishing build graph.')
    
    return G

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


def metis_partition_groups_seeds(G, maximum_seed_size):
    print('Partitioning...')
    if type(G) is StellarGraph:
        G = nx.Graph(G.to_networkx())

    CC = [cc for cc in nx.connected_components(G)]
    GL = []
    for subV in CC:
        if len(subV) > maximum_seed_size:
            # use metis to split the graph
            subG = nx.subgraph( G, subV )
            nparts = int( len(subV)/maximum_seed_size + 1 )
            ( edgecuts, parts ) = nxmetis.partition( subG, nparts, edge_weight='weight' )
            
            # only add connected components
            for p in parts:
                pG = nx.subgraph( G, p )
                GL += [list(cc) for cc in nx.connected_components( pG )]
            
            # add to group list
            #GL += parts
        else:
            GL += [list(subV)]

    SL = []
    for p in GL:
        pG = nx.subgraph( G, p )
        SL += [nx.maximal_independent_set( pG )]

    def long_int2int(xs):
        return [np.uint32(x).item() for x in xs]

    GL = [long_int2int(l) for l in GL]
    SL = [long_int2int(l) for l in SL]
    return GL, SL

def metis_partition_groups_seeds_stellargraph(G, maximum_seed_size):
    import metis
    print('Partitioning stellar graph...')

    # First, acquire connected components
    # CC = G.connected_components() # iterator object
    CC = connected_components(G) # iterator object
    GL = []

    for subV in CC:
        if len(subV) > maximum_seed_size:
            # use metis to split the graph
            subG = G.subgraph(subV)
            nparts = int( len(subV)/maximum_seed_size + 1 )
            lil_adj = subG.to_adjacency_matrix(weighted=False).tolil()
            adjlist = [tuple(neighbours) for neighbours in lil_adj.rows]

            edgecuts, parts = metis.part_graph(adjlist, nparts=nparts)

            parts = np.array(parts)
            clusters = []
            node_ids = np.array(subG.nodes())
            cluster_ids = np.unique(parts)
            for cluster_id in cluster_ids:
                mask = np.where(parts == cluster_id)
                clusters.append(node_ids[mask].tolist())
            
            parts = clusters
            # only add connected components
            for p in parts:
                pG = G.subgraph(p)
                GL += [list(cc) for cc in pG.connected_components()]
            
            # add to group list
            #GL += parts
        else:
            GL += [list(subV)]

    SL = []
    for p in GL:
        pG = G.subgraph(p)
        SL += [nx.maximal_independent_set(pG.to_networkx())]

    def long_int2int(xs):
        return [np.uint32(x).item() for x in xs]

    GL = [long_int2int(l) for l in GL]
    SL = [long_int2int(l) for l in SL]
    return GL, SL

def plain_bfs(adj, node):
    seen = set()
    nextlevel = {node}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(adj[v])
    
    return seen

def connected_components(G):
    nodes = G.nodes()
    lil_adj = G.to_adjacency_matrix(weighted=False).tolil()
    adjlist = [tuple(neighbours) for neighbours in lil_adj.rows]
    seen = set()
    for v in nodes:
        if v not in seen:
            c = plain_bfs(adjlist, v)
            seen.update(c)
            yield c


if __name__ == "__main__":
    # amd_file = 'E:\\workspace\\python\\metagenomics-deep-binning\\data\\amd\\Amdfull\\parsed\\blasted_amd.fna'
    # reads, labels = load_amd_reads(amd_file)

    # print([0] * 10)

    # sample_file = 'E:\\workspace\\python\\metagenomics-deep-binning\\data\\simulated\\raw\\L1.fna'
    # load_meta_reads(sample_file)
    # a = 1

    sample_dict = {k:k for k in range(100)}
    splitted_dicts = split_dict(sample_dict, 10)

    print(splitted_dicts)
