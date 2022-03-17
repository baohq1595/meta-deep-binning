import glob, os, time
import json
import argparse
import tqdm
import numpy as np
import wandb

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling

from metadec.dataset.genome import SimGenomeDataset
from metadec.model.dec import DEC
from metadec.model.idec import IDEC
from metadec.model.adec import *
from metadec.utils.utils import load_genomics
from metadec.debug.visualize import store_results
from metadec.utils.metrics import genome_acc

import sys
sys.path.append('.')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, help='directory contain fasta file')
    parser.add_argument('-n', '--dataset_name', type=str, help='specific name of dataset to run'\
                        '(`all` is run all found datasets), e.g S1.fna, name is S1')
    parser.add_argument('-l', '--lmer', type=int, help='Length of parameter lmer')
    parser.add_argument('--n_shared_read', type=int, help='Number of shared reads threshold in building graph')
    parser.add_argument('--seed_size', type=int, help='Maximum seed size')
    parser.add_argument('--n_clusters', type=int, help='Number of clusters')
    parser.add_argument('--phase1_only', action='store_true', help='Generate data and finish, no training')
    parser.add_argument('--load_cache', action='store_true', help='Load phase 1 from cache. Cache file path is data_dir.parent/processed')
    parser.add_argument('--cache', action='store_true', help='Cache phase 1 result. Cache file path is data_dir.parent/processed')
    parser.add_argument('--is_amd', action='store_true', help='Turn on to False for simulated datasets, otherwise, turn off')
    parser.add_argument('--result_dir', type=str, default='results', help='directory for saving the resuls')
    parser.add_argument('--max_iters', type=int, default=1, help='Number of iterations for running cluster phase')
    parser.add_argument('--pretrain_epochs', type=int, default=2000, help='Number of epochs for pretraining phase 1')
    parser.add_argument('--pretrain2_epochs', type=int, default=100, help='Number of epochs for pretraining phase 2')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--verbose', help='Whether to print log in terminal', action='store_true')

    args = parser.parse_args()

    verbose = args.verbose

    # Maximum iterations for clustering optimization step
    MAX_ITERS = args.max_iters
    # Number of epochs for pretraining
    PRETRAIN_EPOCHS = args.pretrain_epochs
    # PRETRAIN_EPOCHS = 5000 # for s8, s9, s10
    PRETRAIN_PHASE2_EPOCHS = args.pretrain2_epochs
    DATASET_NAME = args.dataset_name
    GEN_DATA_DIR = args.data_dir
    RESULT_DIR = args.result_dir
    BATCH_SIZE = args.batch_size
    TOL = 0.00001
    data_dir = args.data_dir
    num_shared_read = args.n_shared_read
    lmers = args.lmer
    max_seed_size = args.seed_size
    is_amd = args.is_amd

    # Hyperparameters
    # Follows metaprob
    KMERS = [4]
    ONLY_SEED = True

    processed_dir = os.path.join(data_dir, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if DATASET_NAME == 'all':
        raw_datasets = glob.glob(data_dir + '/*.fna')
    else:
        raw_datasets = [os.path.join(data_dir, DATASET_NAME + '.fna')]

    for dataset in tqdm.tqdm(raw_datasets):
        # Get some parameters
        dataset_name = os.path.basename(dataset).split('.fna')[0]
        save_dir = os.path.join(RESULT_DIR, dataset_name)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        is_deserialize = os.path.exists(os.path.join(processed_dir, dataset_name + '.json')) and args.load_cache
        is_serialize = args.cache
        n_clusters = args.n_clusters
        
        # Read dataset
        print(f'Processing dataset {dataset_name}...')
        print(f'Prior number of clusters: {n_clusters}...')
        print(f'Prior number of shared reads: {num_shared_read}...')

        genome_dataset = load_genomics(
            dataset,
            kmers=KMERS,
            lmer=lmers,
            maximum_seed_size=max_seed_size,
            num_shared_reads=num_shared_read,
            is_deserialize=is_deserialize,
            is_serialize=is_serialize,
            is_normalize=True,
            only_seed=ONLY_SEED,
            graph_file=os.path.join(processed_dir, dataset_name + '.json'),
            is_amd=is_amd
        )

        seed_kmer_features, labels, groups, seeds, label2idx = genome_dataset.kmer_features, genome_dataset.labels,\
                                                                genome_dataset.groups, genome_dataset.seeds,\
                                                                genome_dataset.label2idx

        if args.phase1_only:
            return
        
        base_arch = [136,500,10]
        disc_arch = [136,500,1]
        drop = 0.0
        l_coef = 0.5

        n_clusters = len(np.unique(labels).tolist())
        kwargs = {
            'enc_act': tf.nn.relu,
            'dec_act': tf.nn.relu,
            'dec_out_act': None,
            'critic': True
        }
        
        # for i in tqdm.tqdm_notebook(range(1)):
        adec = ADEC(n_clusters=n_clusters,
                ae_dims=base_arch,
                lambda_coef=l_coef,
                critic_dims=base_arch,
                discriminator_dims=disc_arch,
                dropout=drop,
                tol=TOL,
                **kwargs)

        real_batch_size = BATCH_SIZE if seed_kmer_features.shape[0] > BATCH_SIZE else seed_kmer_features.shape[0]

        initial_lr = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                        initial_lr,
                                        decay_steps=1000*(seed_kmer_features.shape[0] // real_batch_size),
                                        decay_rate=0.90,
                                        staircase=True)
        adec.ae_optim = Adam(initial_lr)
        adec.critic_optim = Adam(initial_lr)
        adec.cluster_optim = Adam(0.0001, epsilon=1e-8)

        print('Model cluster: ', adec.n_clusters)
        wandb.init(project="adec-gene-tf-exp-monitor-time-rerun", config={
            'name': dataset_name,
            'n_clusters': n_clusters,
            'n_shared_reads': num_shared_read,
            'lmer': lmers,
            'max_seeds_size': max_seed_size,
            'arch': base_arch,
            'lr': initial_lr,
            'lambda': l_coef
        })
        
        pretrain(model=adec,
            seeds=seed_kmer_features,
            groups=groups,
            label=labels,
            batch_size=real_batch_size,
            epochs=PRETRAIN_EPOCHS,
            save_interval=PRETRAIN_EPOCHS // 50,
            save_path=f'./{dataset_name}/pretrain/images')

        adec.save('./pretrain_weights')
        
        pretrain_phase2(model=adec,
            seeds=np.array(seed_kmer_features),
            groups=groups,
            label=labels,
            batch_size=real_batch_size,
            epochs=PRETRAIN_PHASE2_EPOCHS,
            save_interval=PRETRAIN_PHASE2_EPOCHS // 50,
            save_path=f'./{dataset_name}/pretrain2/images')

        adec.save('./pretrain_weights')
        # adec.cluster_optim = SGD(learning_rate=0.0001, momentum=0.9)
        # adec.load('./pretrain_weights')
        cluster(model=adec,
            seeds=np.array(seed_kmer_features),
            groups=groups,
            label=labels,
            batch_size=real_batch_size,
            epochs=MAX_ITERS,
            save_interval=4,
            auxiliary_interval=2,
            save_path=f'./{dataset_name}/cluster/images', dataset_name=dataset_name)

if __name__ == "__main__":
    main()
