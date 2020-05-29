from metadec.dataset.genome import SimGenomeDataset

def load_genomics(dataset_name,
                    kmers, 
                    lmer,
                    maximum_seed_size,
                    num_shared_reads,
                    graph_file=None,
                    is_serialize=False,
                    is_deserialize=False,
                    is_normalize=False,
                    only_seed=False):
    '''
    Loads fna file.
    Args:
        dataset_name: name of dataset (e.g. S1.fna, L1.fna,...)
        kmers: list of kmers.
        lmer: lmer.
        maximum_seed_size.
        num_shared_reads.
        graph_file: computed groups/seeds json file.
        is_serialize: True to serialize computed groups/seeds to json file.
        is_deserialize: True to load computed groups/seeds in json file.
        is_normalize: whether to normalize kmer-features.
        only_seed: True to compute kmer features using seeds only.
    '''
    genomics_dataset = SimGenomeDataset(
        dataset_name, kmers, lmer,
        graph_file=graph_file,
        only_seed=only_seed,
        maximum_seed_size=maximum_seed_size,
        num_shared_reads=num_shared_reads,
        is_serialize=is_serialize,
        is_deserialize=is_deserialize,
        is_normalize=is_normalize)

    return genomics_dataset.kmer_features,\
        genomics_dataset.labels,\
        genomics_dataset.groups,\
        genomics_dataset.seeds