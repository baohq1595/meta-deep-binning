import json
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from metadec.dataset.utils import *

class SimGenomeDataset():
    '''
    Metagenomics dataset for reading simulated data in fasta format (.fna)
    An optimization step based on graph opertation is used to merge reads that
    have overlapping reads into a completed genome.
    '''
    def __init__(self, fna_file, kmers: list, qmers, num_shared_reads,
                     maximum_seed_size=5000, only_seed=False, is_normalize=True,
                     graph_file=None, is_serialize=False, is_deserialize=False):
        '''
        Args:
            kmers: a list of kmer values. 
            fna_file: path to fna file (fasta format).
            only_seed: only seeds in overlapping graph are used to build features.
            graph_file: calculated groups and seeds (json).
        '''
        # Read fasta dataset
        print('Reading fna file...')
        self.reads, self.labels = load_meta_reads(fna_file, type='fasta')

        print('Creating document from reads...')
        dictionary, documents = create_document(self.reads, kmers)

        print('Creating corpus...')
        corpus = create_corpus(dictionary, documents)

        self.groups = []
        self.seeds = []

        if is_deserialize:
            print('Deserializing data...')
            self.groups, self.seeds = self.deserialize_data(graph_file, self.reads)
        else:
            # Build overlapping (reads) graph
            print('Building graph from scratch...')
            graph = build_overlap_graph(self.reads, self.labels, qmers, num_shared_reads=num_shared_reads)
            print('Partitioning graph...')
            self.groups, self.seeds = metis_partition_groups_seeds(graph, maximum_seed_size)

                
        print('Computing features...')
        self.kmer_features = compute_kmer_dist(dictionary, corpus, self.groups, self.seeds, only_seed=only_seed)
        del dictionary
        del corpus
        del self.reads

        if is_serialize:
            print('Serializing data to...', graph_file)
            self.serialize_data(self.groups, self.seeds, graph_file)

        if is_normalize:
            print('Normalizing...')
            scaler = StandardScaler()
            self.kmer_features = scaler.fit_transform(self.kmer_features)

        print('Finish.')
    
    def serialize_data(self, groups, seeds, graph_file):
        serialize_dict = {
            'groups': groups,
            'seeds': seeds
        }

        with open(graph_file, 'w') as fg:
            json.dump(serialize_dict, fg)
        
        return graph_file
    
    def deserialize_data(self, graph_file, reads):
        with open(graph_file, 'r') as fg:
            data = json.load(fg)

        groups = data['groups']
        seeds = data['seeds']

        return groups, seeds