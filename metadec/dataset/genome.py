import imp
import json
import os, pickle
from django import conf
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from compress_pickle import dump as dump_pickle
from metadec.dataset.utils import *

def read_bimeta_cache(filename):
    def parse_file_helper(filename):
        ids_list = []
        i = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if line != '']

            for line in lines:
                # print(line.split(','))
                # exit(1)
                try:
                    ids = list(map(lambda x: int(x)-1, line.split(',')))
                    i += len(ids)
                    ids_list.append(ids)
                except:
                    print(line)

        print(f'Read file {filename}. Found {i} ids.')        
        return ids_list

    groups = parse_file_helper(filename)
    seeds = parse_file_helper(filename.replace('group', 'seed'))
    
    return groups, seeds, {}

class SimGenomeDataset():
    '''
    Metagenomics dataset for reading simulated data in fasta format (.fna)
    An optimization step based on graph opertation is used to merge reads that
    have overlapping reads into a completed genome.
    '''
    def __init__(self, fna_file, kmers: list, qmers, num_shared_reads,
                     maximum_seed_size=5000, only_seed=False, is_normalize=True,
                     graph_file=None, is_serialize=False, is_deserialize=False, is_tfidf=False,
                     serialize_from_bimeta=False):
        '''
        Args:
            kmers: a list of kmer values. 
            fna_file: path to fna file (fasta format).
            only_seed: only seeds in overlapping graph are used to build features.
            graph_file: calculated groups and seeds (json).
        '''
        # Read fasta dataset
        self.serialize_from_bimeta = serialize_from_bimeta
        self.reads, self.labels, self.label2idx = load_meta_reads(fna_file, type='fasta')

        # Creating document from reads...
        dictionary, documents = create_document(self.reads, kmers)

        # Creating corpus...
        corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf)

        self.groups = []
        self.seeds = []

        if is_deserialize:
            # Deserializing data...
            self.groups, self.seeds, label2idx = self.deserialize_data(graph_file)
            self.reads = None
            if len(label2idx) != 0:
                self.label2idx = label2idx
        else:
            # Build overlapping (reads) graph
            # graph = build_overlap_graph(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads)
            # graph = build_overlap_graph_low_mem(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads, parts=20, comp='gzip')
            graph = build_overlap_stellar_graph_low_mem(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads, parts=2, comp='gzip')
            # Partitioning graph...
            self.groups, self.seeds = metis_partition_groups_seeds_stellargraph(graph, maximum_seed_size)
            # self.groups, self.seeds = metis_partition_groups_seeds(graph, maximum_seed_size)

        # with open(os.path.join('temp', 'corpus'), 'rb') as f:
        #     corpus = pickle.load(f)

        # with open(os.path.join('temp', 'dictionary'), 'rb') as f:
        #     dictionary = pickle.load(f)

        # with open(os.path.join('temp', 'documents'), 'rb') as f:
        #     documents = pickle.load(f)
                
        # Computing features...
        self.kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=only_seed)
        del dictionary
        del corpus
        del self.reads

        if is_serialize:
            print('Serializing data to...', graph_file)
            self.serialize_data(self.groups, self.seeds, self.label2idx, graph_file)

        if is_normalize:
            # Normalizing...
            scaler = StandardScaler()
            self.kmer_features = scaler.fit_transform(self.kmer_features)
        
        from compress_pickle import dump as dump_pickle
        with open(os.path.join('temp', 'kmer_features'), 'wb') as f:
            #pickle.dump(self.kmer_features, f)
            dump_pickle(self.kmer_features, f, compression='gzip', set_default_extension=False)
    
    def serialize_data(self, groups, seeds, label2idx, graph_file):
        '''
        Save groups and seeds id to json file
        '''
        serialize_dict = {
            'groups': groups,
            'seeds': seeds,
            'label2idx': label2idx
        }

        if 'json' not in graph_file:
            graph_file = graph_file.split('.')[0] + '.json'

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
    
    def deserialize_data(self, graph_file):
        '''
        Read groups and seeds from file
        '''
        if self.serialize_from_bimeta:
            groups, seeds, label2idx = read_bimeta_cache(graph_file)
        else:
            with open(graph_file, 'r') as fg:
                data = json.load(fg)

            groups = data['groups']
            seeds = data['seeds']
            label2idx = data['label2idx']

        return groups, seeds, label2idx


class AMDGenomeDataset():
    '''
    Metagenomics dataset for reading AMD (acid mine drainage) data
    An optimization step based on graph opertation is used to merge reads that
    have overlapping reads into a completed genome.
    '''
    def __init__(self, fna_file, kmers: list, qmers, num_shared_reads,
                     maximum_seed_size=5000, only_seed=False, is_normalize=True,
                     graph_file=None, is_serialize=False, is_deserialize=False, is_tfidf=False,
                     serialize_from_bimeta=False):
        '''
        Args:
            kmers: a list of kmer values. 
            fna_file: path to fna file (fasta format).
            only_seed: only seeds in overlapping graph are used to build features.
            graph_file: calculated groups and seeds (json).
        '''
        # Read fasta dataset
        self.serialize_from_bimeta = serialize_from_bimeta
        self.reads, self.labels, self.label2idx = load_amd_reads(fna_file)

        print(f'Found {len(self.reads)} reads')

        # Creating document from reads...
        if not os.path.exists(os.path.join('temp', 'corpus')):
            dictionary, documents = create_document(self.reads, kmers)

            #( Creating corpus...
            corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf)

            if not os.path.exists('temp'):
                os.makedirs('temp')

            #with open(os.path.join('temp', 'corpus'), 'wb') as f:
            #    pickle.dump(corpus, f)

            #with open(os.path.join('temp', 'documents'), 'wb') as f:
            #    pickle.dump(documents, f)

            #with open(os.path.join('temp', 'dictionary'), 'wb') as f:
            #    pickle.dump(dictionary, f)

            #del corpus, dictionary, documents
            self.reads = None
            documents = None

        self.groups = []
        self.seeds = []

        if is_deserialize:
            # Deserializing data...
            self.groups, self.seeds, label2idx = self.deserialize_data(graph_file)
            if len(label2idx) != 0:
                self.label2idx = label2idx
        else:
            # Build overlapping (reads) graph
            # graph = build_overlap_graph_low_mem(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads, parts=20, comp='gzip')
            graph = build_overlap_stellar_graph_low_mem(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads, parts=20, comp='gzip')
            # Partitioning graph...
            self.groups, self.seeds = metis_partition_groups_seeds_stellargraph(graph, maximum_seed_size)
                
        # Computing features...
        #with open(os.path.join('temp', 'corpus'), 'rb') as f:
        #    corpus = pickle.load(f)

        #with open(os.path.join('temp', 'dictionary'), 'rb') as f:
        #    dictionary = pickle.load(f)

        #with open(os.path.join('temp', 'documents'), 'rb') as f:
        #    documents = pickle.load(f)
        self.kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=only_seed)
        del dictionary
        del corpus
        # del self.reads

        if is_serialize:
            print('Serializing data to...', graph_file)
            self.serialize_data(self.groups, self.seeds, self.label2idx, self.labels, graph_file)

        if is_normalize:
            # Normalizing...
            scaler = StandardScaler()
            self.kmer_features = scaler.fit_transform(self.kmer_features)
        
        if is_serialize:
            with open(os.path.join('temp', 'kmer_features'), 'wb') as f:
                pickle.dump(self.kmer_features, f, protocol=4)
                #dump_pickle(self.kmer_features, f, compression='gzip', set_default_extension=False)

    
    def serialize_data(self, groups, seeds, label2idx, labels, graph_file):
        '''
        Save groups and seeds id to json file
        '''
        serialize_dict = {
            'groups': groups,
            'seeds': seeds,
            'labels': labels,
            'label2idx': label2idx
        }

        if 'json' not in graph_file:
            graph_file = graph_file.split('.')[0] + '.json'

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
    
    def deserialize_data(self, graph_file):
        '''
        Read groups and seeds from file
        '''
        if self.serialize_from_bimeta:
            groups, seeds, label2idx = read_bimeta_cache(graph_file)
        else:
            with open(graph_file, 'r') as fg:
                data = json.load(fg)

            groups = data['groups']
            seeds = data['seeds']
            label2idx = data['label2idx']

        return groups, seeds, label2idx


class DatasetConfig:
    kmers = [4]
    qmers = 30
    num_shared_reads = 5
    maximum_seed_size = 5000
    only_seed = True
    is_normalize = True
    is_tfidf = False
    cache_bimeta_format = False
    is_amd_format = False
    load_feature_cache = False
    save_feature_cache = False
    load_graph_cache = False
    save_graph_cache = False
    cache_for_graph = ''
    cache_for_feature = ''

    def print():
        print('****Config details****')
        print(f'kmers: {DatasetConfig.kmers}')
        print(f'qmers: {DatasetConfig.qmers}')
        print(f'num_shared_reads: {DatasetConfig.num_shared_reads}')
        print(f'maximum_seed_size: {DatasetConfig.maximum_seed_size}')
        print(f'only_seed: {DatasetConfig.only_seed}')
        print(f'cache_bimeta_format: {DatasetConfig.cache_bimeta_format}')
        print(f'is_amd_format: {DatasetConfig.is_amd_format}')
        print(f'load_feature_cache: {DatasetConfig.load_feature_cache}')
        print(f'load_graph_cache: {DatasetConfig.load_graph_cache}')
        print(f'save_feature_cache: {DatasetConfig.save_feature_cache}')
        print(f'save_graph_cache: {DatasetConfig.save_graph_cache}')



class BaseDataset:
    def __init__(self, fna_file, config: DatasetConfig):
        graph_path = config.cache_for_graph
        feat_path = config.cache_for_feature

        self.groups = None
        self.seeds = None
        self.labels = None
        self.label2idx = None
        self.kmer_feature = None

        # We have everything necessary for running form cache
        if config.load_feature_cache != '':
            if not os.path.exists(graph_path) or not os.path.exists(feat_path):
                raise Exception('Cache path not found, either for graph or feature.')
            # load groups, seeds from cache file
            with open(graph_path, 'r') as f:
                self.groups, self.seeds, self.labels, self.label2idx = self._read_graph_cache(graph_path, config.cache_bimeta_format)
            # load kmer feature from pickle
            with open(feat_path, 'rb') as f:
                self.kmer_feature = pickle.load(f, protocol=4)
        # Only cache for groups, seeds is available
        else:
            self.reads, self.labels, self.label2idx = self.load_read_file(fna_file)

            if config.load_graph_cache:
                with open(graph_path, 'r') as f:
                    self.groups, self.seeds, self.labels, self.label2idx = self._read_graph_cache(graph_path, config.cache_bimeta_format)
            else:
                # Build overlapping (reads) graph
                graph = build_overlap_graph(self.reads, self.labels, config.qmers, num_shared_reads=config.num_shared_reads)
                # Partitioning graph...
                self.groups, self.seeds = metis_partition_groups_seeds(graph, config.maximum_seed_size)

            self.kmer_feature = self._build_feature(config)

        # check if save cache is on, if yes, save cache
        if config.save_graph_cache:
            self.save_graph_cache(self.groups, self.seeds, self.labels, self.label2idx, graph_path)

        if config.save_feature_cache:
            with open(feat_path, 'wb') as f:
                pickle.dump(self.kmer_features, f, protocol=4)

    def save_graph_cache(self, groups, seeds, label2idx, labels, graph_file):
        '''
        Save groups and seeds id to json file
        '''
        serialize_dict = {
            'groups': groups,
            'seeds': seeds,
            'labels': labels,
            'label2idx': label2idx
        }

        if 'json' not in graph_file:
            graph_file = graph_file.split('.')[0] + '.json'

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
            

    def load_read_file(self, path):
        raise NotImplementedError('Should be inherited!')

    def _build_feature(self, config: DatasetConfig):
        dictionary, documents = create_document(self.reads, config.kmers)

        # Creating corpus...
        corpus = create_corpus(dictionary, documents, is_tfidf=config.is_tfidf)
        documents = None # free some memory

        kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=config.only_seed)
        # Normalizing...
        if config.is_normalize:
            scaler = StandardScaler()
            kmer_features = scaler.fit_transform(kmer_features)

        return kmer_features


    def _read_graph_cache(self, graph_path, bimeta_format):
        '''
        Read groups and seeds from file
        '''
        if bimeta_format:
            groups, seeds, label2idx = read_bimeta_cache(graph_path)
        else:
            with open(graph_path, 'r') as fg:
                data = json.load(fg)

            groups = data['groups']
            seeds = data['seeds']
            labels = data['labels']
            label2idx = data['label2idx']

        return groups, seeds, labels, label2idx

class SimulatedDataset(BaseDataset):
    def __init__(self, fna_file, config: DatasetConfig):
        super().__init__(fna_file, config)

    def load_read_file(self, path):
        reads, labels, label2idx = load_meta_reads(path, type='fasta')
        return reads, labels, label2idx

class RealDataset(BaseDataset):
    def __init__(self, fna_file, config: DatasetConfig):
        super().__init__(fna_file, config)

    def load_read_file(self, path):
        reads, labels, label2idx = load_amd_reads(path)
        return reads, labels, label2idx