import json
import os, pickle
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from metadec.dataset.utils import *
import time
class SimGenomeDataset():
    '''
    Metagenomics dataset for reading simulated data in fasta format (.fna)
    An optimization step based on graph opertation is used to merge reads that
    have overlapping reads into a completed genome.
    '''
    def __init__(self, fna_file, kmers: list, qmers, num_shared_reads,
                     maximum_seed_size=5000, only_seed=False, is_normalize=True,
                     graph_file=None, is_serialize=False, is_deserialize=False, is_tfidf=False, n_procs=1):
        '''
        Args:
            kmers: a list of kmer values. 
            fna_file: path to fna file (fasta format).
            only_seed: only seeds in overlapping graph are used to build features.
            graph_file: calculated groups and seeds (json).
        '''
        # Read fasta dataset
        self.reads, self.labels = load_meta_reads(fna_file, type='fasta')

        # Creating document from reads...
        dictionary, documents = create_document(self.reads, kmers)

        # Creating corpus...
        corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf)

        self.groups = []
        self.seeds = []

        if is_deserialize:
            # Deserializing data...
            self.groups, self.seeds = self.deserialize_data(graph_file, self.reads)
        else:
            # Build overlapping (reads) graph
            start = time.time()
            graph = build_overlap_graph(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads)
            print("Time for build graph: ", time.time()-start)
            # Partitioning graph...
            self.groups, self.seeds = metis_partition_groups_seeds(graph, maximum_seed_size)
                
        # Computing features...
        self.kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=only_seed)
        del dictionary
        del corpus
        del self.reads

        if is_serialize:
            print('Serializing data to...', graph_file)
            self.serialize_data(self.groups, self.seeds, graph_file)

        if is_normalize:
            # Normalizing...
            scaler = StandardScaler()
            self.kmer_features = scaler.fit_transform(self.kmer_features)
    
    def serialize_data(self, groups, seeds, graph_file):
        '''
        Save groups and seeds id to json file
        '''
        serialize_dict = {
            'groups': groups,
            'seeds': seeds
        }

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
    
    def deserialize_data(self, graph_file, reads):
        '''
        Read groups and seeds from file
        '''
        with open(graph_file, 'r') as fg:
            data = json.load(fg)

        groups = data['groups']
        seeds = data['seeds']

        return groups, seeds


class AMDGenomeDataset():
    '''
    Metagenomics dataset for reading AMD (acid mine drainage) data
    An optimization step based on graph opertation is used to merge reads that
    have overlapping reads into a completed genome.
    '''
    def __init__(self, fna_file, kmers: list, qmers, num_shared_reads,
                     maximum_seed_size=5000, only_seed=False, is_normalize=True,
                     graph_file=None, is_serialize=False, is_deserialize=False, is_tfidf=False, n_procs=1):
        '''
        Args:
            kmers: a list of kmer values. 
            fna_file: path to fna file (fasta format).
            only_seed: only seeds in overlapping graph are used to build features.
            graph_file: calculated groups and seeds (json).
        '''
        # Read fasta dataset
        print('Read dataset...')
        self.reads, self.labels = load_amd_reads(fna_file)
        basename = os.path.basename(fna_file).split('.')[0]

        # Creating document from reads...
        if not os.path.exists(os.path.join('temp/basename', 'corpus')):
            dictionary, documents = create_document(self.reads, kmers)

            # Creating corpus...
            print('Creating corpus...')
            corpus = create_corpus(dictionary, documents, is_tfidf=is_tfidf)

            if not os.path.exists('temp/basename'):
                os.makedirs('temp/basename')

            with open(os.path.join('temp/basename', 'corpus'), 'wb') as f:
                pickle.dump(corpus, f)

            with open(os.path.join('temp/basename', 'documents'), 'wb') as f:
                pickle.dump(documents, f)

            with open(os.path.join('temp', 'dictionary'), 'wb') as f:
                pickle.dump(dictionary, f)

            del corpus, dictionary, documents

        self.groups = []
        self.seeds = []

        if is_deserialize:
            # Deserializing data...
            self.groups, self.seeds = self.deserialize_data(graph_file, self.reads)
        else:
            # Build overlapping (reads) graph
            graph = build_overlap_graph(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads)#, n_procs=n_procs)
            # Partitioning graph...
            self.groups, self.seeds = metis_partition_groups_seeds(graph, maximum_seed_size)
                
        # Computing features...
        with open(os.path.join('temp/basename', 'corpus'), 'rb') as f:
            corpus = pickle.load(f)

        with open(os.path.join('temp/basename', 'dictionary'), 'rb') as f:
            dictionary = pickle.load(f)

        with open(os.path.join('temp/basename', 'documents'), 'rb') as f:
            documents = pickle.load(f)
        self.kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=only_seed)
        del dictionary
        del corpus
        # del self.reads

        if is_serialize:
            print('Serializing data to...', graph_file)
            self.serialize_data(self.groups, self.seeds, graph_file)

        if is_normalize:
            # Normalizing...
            scaler = StandardScaler()
            self.kmer_features = scaler.fit_transform(self.kmer_features)
    
    def serialize_data(self, groups, seeds, graph_file):
        '''
        Save groups and seeds id to json file
        '''
        serialize_dict = {
            'groups': groups,
            'seeds': seeds
        }

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
    
    def deserialize_data(self, graph_file, reads):
        '''
        Read groups and seeds from file
        '''
        with open(graph_file, 'r') as fg:
            data = json.load(fg)

        groups = data['groups']
        seeds = data['seeds']

        return groups, seeds
