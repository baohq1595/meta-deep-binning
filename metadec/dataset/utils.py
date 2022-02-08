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
import os, sys
import math

from metadec.utils.hashtable import simple_str2numeric, Hashtable

def load_amd_reads(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    reads = []
    raw_labels = []
    labels = []
    is_read_part = False
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
    label2idx = {label: i for i, label in enumerate(unique_label)}
    labels = [label2idx[label] for label in raw_labels]
    
    return reads, labels


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

        return reads, labels
    except Exception as e:
        print('Error when loading file {} '.format(filename))
        print('Cause: ', e)
        return []

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
    print('Start computing feature')
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


def build_overlap_graph(reads, labels, qmer_length, num_shared_reads, hash_size):
    '''
    Build overlapping graph
    '''
    # Create hash table with q-mers are keys
    # lmers_dict=dict()
    print('Build hashtable')
    lmers_dict = Hashtable(hash_size)
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            lmer = simple_str2numeric(r[j:j+qmer_length])
            if lmers_dict.get(lmer, default_value=None) is None:
                lmers_dict.insert(lmer, [idx])
            else:
                lmers_dict.get(lmer, []).append(idx)

    print('Start initializing graph')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes')
    # Add nodes to graph
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    # Add edges to graph
    print('Add edges')
    for i in range(lmers_dict.size):
        seq_id_list = lmers_dict.hashmap[i]
        if seq_id_list is None:
            continue
        for e in it.combinations(seq_id_list,2):
            if e[0] != e[1]:
                if G.has_edge(**e):
                    G[e[0]][e[1]]['weight'] += 1
                else:
                    G.add_edge(e[0], e[1], weight=1)
            else:
                continue

    # Finishing....
    print('Finishing build graph')
    
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

def build_overlap_graph_v2(reads, labels, qmer_length, num_shared_reads):
    '''
    Build overlapping graph
    '''
    if not os.path.exists('temp'):
        os.makedirs('temp')

    for i, r in enumerate(reads):
        reads[i].replace('N', '')

    # Create hash table with q-mers are keys
    lmers_dict=dict()
    for idx, r in enumerate(reads):
        for j in range(0,len(r)-qmer_length+1):
            lmer = r[j:j+qmer_length]
            if lmer in lmers_dict:
                lmers_dict[lmer] += [idx]
            else:
                lmers_dict[lmer] = [idx]

    # Building edges
    # E=dict()
    chunks = len(lmers_dict)
    splitted_dicts = split_dict(lmers_dict, chunks)
    print('Building...')
    for i, sub_lmers_dict in enumerate(splitted_dicts):
        E=dict()
        count = 0
        for lmer in sub_lmers_dict:
            for e in it.combinations(sub_lmers_dict[lmer],2):
                if e[0]!=e[1]:
                    e_curr=(e[0],e[1])
                else:
                    continue
                if e_curr in E:
                    E[e_curr] += 1 # Number of connected lines between read a and b
                else:
                    E[e_curr] = 1
                count += 1
            
        with open(os.path.join('temp', f'chunk{i}'), 'wb') as f:
            pickle.dump(E, f)
        del E
        splitted_dicts[i] = None

    print('Filtering...')

    E = dict()
    for i in range(chunks):
        with open(os.path.join('temp', f'chunk{i}'), 'rb') as f:
            sub_E = pickle.load(f)

        for kv in sub_E.items():
            if kv[0] in E:
                E[kv[0]] += sub_E[kv[0]]
            else:
                E[kv[0]] = 1
    
    print('Finish building edges.')
        
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}

    print('Start initializing graph')
    # Initialize graph
    G = nx.Graph()
    
    # Add nodes to graph
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'darkcyan', 5: 'violet'}
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

    # Finishing....
    
    return G

def metis_partition_groups_seeds(G, maximum_seed_size):
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

    return GL, SL

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
