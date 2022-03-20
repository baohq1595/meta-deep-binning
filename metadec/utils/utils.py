import json
from metadec.dataset.genome import SimGenomeDataset, AMDGenomeDataset, DatasetConfig,\
    SimulatedDataset, RealDataset

def load_genomics(dataset_name,
                    kmers, 
                    lmer,
                    maximum_seed_size,
                    num_shared_reads,
                    graph_file=None,
                    is_serialize=False,
                    is_deserialize=False,
                    is_normalize=False,
                    only_seed=False,
                    is_tfidf=False,
                    is_amd=False,
                    serialize_from_bimeta=False):
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
    if not is_amd:
        genomics_dataset = SimGenomeDataset(
            dataset_name, kmers, lmer,
            graph_file=graph_file,
            only_seed=only_seed,
            maximum_seed_size=maximum_seed_size,
            num_shared_reads=num_shared_reads,
            is_serialize=is_serialize,
            is_deserialize=is_deserialize,
            is_normalize=is_normalize,
            is_tfidf=is_tfidf,
            serialize_from_bimeta=serialize_from_bimeta)
    else:
        genomics_dataset = AMDGenomeDataset(
            dataset_name, kmers, lmer,
            graph_file=graph_file,
            only_seed=only_seed,
            maximum_seed_size=maximum_seed_size,
            num_shared_reads=num_shared_reads,
            is_serialize=is_serialize,
            is_deserialize=is_deserialize,
            is_normalize=is_normalize,
            is_tfidf=is_tfidf,
            serialize_from_bimeta=serialize_from_bimeta)

    return genomics_dataset

def load_dataset(dataset_path, dataset_config):
    if dataset_config.is_amd_format:
        dataset = RealDataset(dataset_path, dataset_config)
    else:
        dataset = SimulatedDataset(dataset_path, dataset_config)

    return dataset

def export_clustering_results(raw_reads, groups, n_clusters, y_pred, save_path):
    exported_results = {k+1: [] for k in range(n_clusters)}

    for i, group in enumerate(groups):
        cluster_id = y_pred[i]
        for r in group:
            exported_results[cluster_id + 1].append(r)
    
    with open(save_path, 'w') as f:
        json.dump(exported_results, f)

    print(f'Saved result file at {save_path}')


