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
import multiprocessing
import json
import pickle

alphabet = 'abcdefghijklmnopqrvuxyzw'
special_chars = '.,;:-='

bases_4 = ['A', 'C', 'G', 'T']
kmers_map = {
    4: [''.join(p) for p in it.product(bases_4, repeat=4)],
    3: [''.join(p) for p in it.product(bases_4, repeat=3)],
    2: [''.join(p) for p in it.product(bases_4, repeat=2)],
    1: [''.join(p) for p in it.product(bases_4, repeat=1)]
}
# print(kmers_map)
merged_kmer_list = [x for kv in kmers_map.items() for x in kv[1]]

def encode_hash(hash_str):
    chars = []
    str_part_size = int(math.floor(len(hash_str) / 4))
    remains = len(hash_str) - str_part_size * 4
    offset = 0
    for i in range(str_part_size):
        x_mer = hash_str[offset: offset + 4]
        idx = kmers_map[4].index(x_mer)
        chars.append(str(idx))

def hash2numeric(hash_str):
    '''
    A: 00
    T: 01
    G: 10
    C: 11
    '''
    val = 1
    for nucl in hash_str:
        val = val << 2
        if nucl == 'A':
            val |= 0
        elif nucl == 'T':
            val |= 1
        elif nucl == 'G':
            val |= 2
        elif nucl == 'C':
            val |= 3
        else:
            raise Exception('Bad nucliotide character!!!')

    return val

def numeric2hash(numeric_val, expected_length=30):
    chars = []
    while(numeric_val > 1):
        mod = numeric_val % 4
        numeric_val = numeric_val // 4
        if mod == 3:
            char = 'C'
        elif mod == 2:
            char = 'G'
        elif mod == 1:
            char = 'T'
        elif mod == 0:
            char = 'A'
        else:
            raise Exception('Bad numeric value!!!')
        chars.append(char)

    if len(chars) != expected_length:
        raise Exception('Decoded length does not match with expected length!!!')

    return ''.join(list(reversed(chars)))


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


def build_overlap_graph(reads, labels, qmer_length, num_shared_reads):
    '''
    Build overlapping graph
    '''
    # Create hash table with q-mers are keys
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
    print('Start initializing graph')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes')
    
    # Add nodes to graph
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'darkcyan', 5: 'violet'}
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    print('Add edges')

    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

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

def build_hashtable_worker(sub_reads, qmer_length, temp_dir, id, offset):
    # Create hash table with q-mers are keys
    sub_lmers_dict = dict()
    print('Sub Reads: ', len(sub_reads))
    for idx, r in enumerate(sub_reads, start=offset):
        for j in range(0,len(r)-qmer_length+1):
            lmer = r[j:j+qmer_length]
            encoded_lmer = hash2numeric(lmer)
            # encoded_lmer = lmer
            if encoded_lmer in sub_lmers_dict:
                sub_lmers_dict[encoded_lmer] += [idx]
            else:
                sub_lmers_dict[encoded_lmer] = [idx]

    # with open(os.path.join(temp_dir, f'hash_dict_{id}'), 'w') as f:
    #     json.dump(sub_lmers_dict, f)
    with open(os.path.join(temp_dir, f'hash_dict_{id}'), 'wb') as f:
        pickle.dump(sub_lmers_dict, f)
    print('Worker finish')

def str2tuple(s):
    tup = []
    for it in re.finditer(r'[0-9]+', s):
        tup.append(int(it.group(0)))
    return tuple(tup)

def build_edges_worker(lmers_dict, temp_dir, id):
    e_dict = dict()
    print(f'Start {id}')
    for encoded_lmer in lmers_dict:
        for e in it.combinations(lmers_dict[encoded_lmer], 2):
            if e[0]!=e[1]:
                # e_curr=str((e[0],e[1]))
                e_curr=(e[0],e[1])
            else:
                continue
            if e_curr in e_dict:
                e_dict[e_curr] += 1 # Number of connected lines between read a and b
            else:
                e_dict[e_curr] = 1
    
    print(f'End {id}')
    # with open(os.path.join(temp_dir, f'edge_dict_{id}'), 'w') as f:
    #     json.dump(e_dict, f)

    with open(os.path.join(temp_dir, f'edge_dict_{id}'), 'wb') as f:
        pickle.dump(e_dict, f)

    print('Worker finish')


def combine_e_dicts(e_dicts):
    e_dict = e_dicts[0]
    for d in e_dicts[1:]:
        for conn, count in d.items():
            if conn in e_dict:
                e_dict[conn] += count
            else:
                e_dict[conn] = count

    return e_dict

def combine_lmers_dicts(lmers_dicts):
    lmers_dict = lmers_dicts[0]
    for d in lmers_dicts[1:]:
        for lmer, idxs in d.items():
            if lmer in lmers_dict:
                lmers_dict[lmer].extend(idxs)
            else:
                lmers_dict[lmer] = idxs

    return lmers_dict


def build_overlap_graph_v2(reads, labels, qmer_length, num_shared_reads):
    '''
    Build overlapping graph
    '''
    for i, r in enumerate(reads):
        reads[i] = reads[i].replace('N', '')
    # Create hash table with q-mers are keys
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
    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    print('Start initializing graph')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes')
    
    # Add nodes to graph
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'darkcyan', 5: 'violet'}
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    print('Add edges')

    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

    # Finishing....
    print('Finishing build graph')
    
    return G

def build_hashtable(reads, qmer_length, save_dir, n_procs=1):
    lmers_dict = dict()
    reads_sublist = split_list(reads, n_procs)

    print('Reads: ', len(reads))
    # temp_dir = 'temp'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    procs = []
    lmers_dicts = []
    shared_mem_dicts = []
    offset = 0
    for i in range(n_procs):
        # sub_lmers_dict = multiprocessing.Manager().dict()
        pi = multiprocessing.Process(target=build_hashtable_worker,
                                    args=[
                                            reads_sublist[i],
                                            qmer_length,
                                            save_dir,
                                            i, offset
                                        ])
        offset += len(reads_sublist[i])
        procs.append(pi)
        # shared_mem_dicts.append(sub_lmers_dict)
        pi.start()
        print('Process bh spawn!')

    for p in procs:
        p.join()

    for i in range(n_procs):
        # with open(os.path.join(temp_dir, f'hash_dict_{i}'), 'r') as f:
        #     lmers_dicts.append(json.load(f))
        with open(os.path.join(save_dir, f'hash_dict_{i}'), 'rb') as f:
            d = pickle.load(f)
            # d = {int(kv[0]): kv[1] for kv in d.items()}
            lmers_dicts.append(d)

    # for d in shared_mem_dicts:
    #     lmers_dicts.append(d.copy())

    lmers_dict = combine_lmers_dicts(lmers_dicts)
    return lmers_dict

def build_edges(lmers_dict, n_procs=1):
    e_dict = dict()
    print('aaa', len(lmers_dict))
    sub_dicts = split_dict(lmers_dict, n_procs)
    print('ccc', len(sub_dicts))
    print('bbb', len(sub_dicts[0]))
    procs = []
    e_dicts = []
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    for i in range(n_procs):
        pi = multiprocessing.Process(target=build_edges_worker,
                                    args=[
                                            sub_dicts[i],
                                            temp_dir,
                                            i
                                        ])
        procs.append(pi)
        pi.start()
        print(f'Process be {i} spawn!')

    for p in procs:
        p.join()

    # for d in shared_mem_dicts:
    #     e_dicts.append(d.copy())
    for i in range(n_procs):
        # with open(os.path.join(temp_dir, f'edge_dict_{i}'), 'r') as f:
        #     d = json.load(f)
        #     d = {str2tuple(kv[0]): kv[1] for kv in d.items()}
        with open(os.path.join(temp_dir, f'edge_dict_{i}'), 'rb') as f:
            d = pickle.load(f)
            # d = {str2tuple(kv[0]): kv[1] for kv in d.items()}

            e_dicts.append(d)

    e_dict = combine_e_dicts(e_dicts)
    return e_dict


def build_overlap_graph_fast(reads, labels, qmer_length, num_shared_reads, n_procs=1, save_dir='temp'):
    '''
    Build overlapping graph
    '''
    for i, r in enumerate(reads):
        reads[i] = reads[i].replace('N', '')
    # Create hash table with q-mers are keys
    lmers_dict = build_hashtable(reads, qmer_length, save_dir=save_dir, n_procs=n_procs)

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

    print('Finish edges')

    E_Filtered = {kv[0]: kv[1] for kv in E.items() if kv[1] >= num_shared_reads}
    print('Start initializing graph')
    # Initialize graph
    G = nx.Graph()

    print('Add nodes')
    
    # Add nodes to graph
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'darkcyan', 5: 'violet'}
    for i in range(0, len(labels)):
        G.add_node(i, label=labels[i])

    print('Add edges')

    # Add edges to graph
    for kv in E_Filtered.items():
        G.add_edge(kv[0][0], kv[0][1], weight=kv[1])

    # Finishing....
    print('Finishing build graph')
    
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

    # sample_dict = {k:k for k in range(100)}
    # splitted_dicts = split_dict(sample_dict, 10)

    # print(splitted_dicts)
    val = hash2numeric('ACTACGATGATCTAGCTGATCTCACGGTTT')
    print(numeric2hash(1152921504623642130))

    # a = (1334, 579, 9459, 1230, 19384,1200939,10231939,1,32,46,7,7,8)
    # print(str2tuple(str(a)))

    